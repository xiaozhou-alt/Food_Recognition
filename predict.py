import os
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import json  # 新增：用于加载类别映射

# 设置随机种子函数
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 在代码开头设置随机种子
set_seed(42)

# 解决 torchaudio 缺少后端的问题
try:
    import soundfile as sf
    torchaudio.set_audio_backend("soundfile")
except:
    print("无法设置 soundfile 后端，尝试使用默认后端")

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据集参数
SAMPLE_RATE = 16000
DURATION = 7000  # 毫秒
N_MELS = 128
TEST_FOLDER = './data/test_a'
MODEL_PATH = './output/model/best_model.pth'
OUTPUT_CSV = './sample_submit.csv'

# 新增：从训练时保存的文件加载类别映射
with open('./output/class_mapping.json', 'r') as f:
    class_mapping = json.load(f)
    idx_to_class = {int(k): v for k, v in class_mapping['idx_to_class'].items()}
    class_to_idx = class_mapping['class_to_idx']

print("✅ 类别映射加载成功")
print(f"类别数量: {len(idx_to_class)}")

# 时域拉伸类（与训练代码一致）
class ComplexTimeStretch:
    def __init__(self, sample_rate, p=0.0):  # 预测时p=0，不使用时域拉伸
        self.time_stretch = T.TimeStretch()
        self.p = p
        self.sample_rate = sample_rate
        self.n_fft = 2048
        self.hop_length = 300
    
    def __call__(self, waveform):
        if torch.rand(1).item() < self.p:
            spec = torch.stft(
                waveform.squeeze(0), 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                win_length=1200,
                return_complex=True
            )
            rate = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
            stretched_spec = self.time_stretch(spec, rate)
            stretched_waveform = torch.istft(
                stretched_spec, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                win_length=1200,
                length=waveform.shape[1]
            )
            return stretched_waveform.unsqueeze(0)
        return waveform

# 频谱增强类（与训练代码一致）
class SpecAugment:
    def __init__(self, freq_mask_param=0, time_mask_param=0, p=0.0):  # 预测时p=0，不使用增强
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        self.p = p
    
    def __call__(self, spec):
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
        
        if torch.rand(1).item() < self.p:
            spec = self.freq_mask(spec)
        
        if torch.rand(1).item() < self.p:
            spec = self.time_mask(spec)
        
        if torch.rand(1).item() < self.p:
            gain = torch.FloatTensor(1).uniform_(-6, 6)
            spec = spec + gain
        
        return spec.squeeze(0)

# 自定义测试数据集类（与训练代码保持一致）
class TestAudioDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_paths = []
        self.time_stretch = ComplexTimeStretch(SAMPLE_RATE, p=0.0)  # 预测时不使用时域拉伸
        self.spec_augment = SpecAugment(p=0.0)  # 预测时不使用频谱增强
        
        # 收集所有音频文件路径
        for file in os.listdir(data_path):
            if file.endswith('.wav'):
                self.file_paths.append(os.path.join(data_path, file))
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        filename = os.path.basename(audio_path)
        
        try:
            waveform, orig_sr = torchaudio.load(audio_path)
        except:
            try:
                import soundfile as sf
                data, orig_sr = sf.read(audio_path)
                waveform = torch.tensor(data).float()
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=1)  # 转换为单声道
                waveform = waveform.unsqueeze(0)  # 添加通道维度 [1, samples]
            except Exception as e:
                print(f"无法加载音频文件 {audio_path}: {e}")
                waveform = torch.zeros(1, int(SAMPLE_RATE * (DURATION / 1000)))
                orig_sr = SAMPLE_RATE
        
        # 重采样
        if orig_sr != SAMPLE_RATE:
            resampler = T.Resample(orig_sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # 统一音频长度
        target_samples = int(SAMPLE_RATE * (DURATION / 1000))
        if waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
        else:
            padding = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # 应用时域拉伸（预测时禁用）
        waveform = self.time_stretch(waveform)
        
        # 转换为梅尔频谱图
        mel_specgram = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=2048,
            win_length=1200,
            hop_length=300,
            n_mels=N_MELS
        )(waveform)
        
        # 转换为分贝单位
        mel_specgram = T.AmplitudeToDB()(mel_specgram)
        
        # 应用频谱增强（预测时禁用）
        mel_specgram = self.spec_augment(mel_specgram)
        
        # 添加通道维度 (1, n_mels, time)
        if mel_specgram.dim() == 2:
            mel_specgram = mel_specgram.unsqueeze(0)
        
        return mel_specgram, filename

# 定义预训练模型（与训练时相同的结构）
class CNN14(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super(CNN14, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # 与训练代码保持一致的处理
        if x.dim() == 5:
            x = x.squeeze(1)  # 从 [B, 1, 1, H, W] -> [B, 1, H, W]
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 预测函数
def predict(model, loader):
    model.eval()
    filenames = []
    predictions = []
    all_outputs = []  # 用于调试
    
    with torch.no_grad():
        for inputs, batch_filenames in tqdm(loader, desc="预测中"):
            # 确保输入维度正确（与训练时一致）
            if inputs.dim() == 5:
                inputs = inputs.squeeze(1)
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            filenames.extend(batch_filenames)
            predictions.extend(preds.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    
    return filenames, predictions, all_outputs

def main():
    # 设置随机种子
    set_seed(42)
    
    # 加载模型
    model = CNN14(num_classes=len(idx_to_class)).to(device)
    
    # 加载模型权重（确保map_location正确）
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ 模型加载成功")
    print(f"模型类别数: {len(idx_to_class)}")

    # 创建测试数据集和数据加载器
    test_dataset = TestAudioDataset(TEST_FOLDER)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )
    print(f"测试集样本数: {len(test_dataset)}")

    # 执行预测
    filenames, pred_indices, all_outputs = predict(model, test_loader)

    # 将预测索引转换为类别名称
    pred_labels = [idx_to_class[idx] for idx in pred_indices]

    # 创建结果DataFrame
    results = pd.DataFrame({
        'name': filenames,
        'label': pred_labels
    })

    # 保存为CSV文件
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 预测结果已保存至: {OUTPUT_CSV}")
    print(f"预测样本数量: {len(results)}")

    # 显示部分结果和置信度
    print("\n预测结果示例:")
    for i in range(min(5, len(results))):
        filename = results.iloc[i]['name']
        label = results.iloc[i]['label']
        probs = torch.softmax(torch.tensor(all_outputs[i]), dim=0)
        top_prob, top_idx = torch.topk(probs, 3)
        top_classes = [idx_to_class[idx.item()] for idx in top_idx]
        print(f"{filename}: {label} (置信度: {top_prob[0].item():.4f})")
        print(f"  其他可能: {', '.join([f'{c} ({p:.4f})' for c, p in zip(top_classes[1:], top_prob[1:])])}")

if __name__ == '__main__':
    main()