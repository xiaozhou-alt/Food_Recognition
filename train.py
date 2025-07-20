import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import IPython.display as ipd
import random

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据集参数
DATA_PATH = '/kaggle/input/eating-sound/data/data/train'
SAMPLE_RATE = 16000
DURATION = 7000  # 毫秒
N_MELS = 128
N_CLASSES = 20

# 获取类别映射
class_names = sorted(os.listdir(DATA_PATH))
class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
idx_to_class = {i: cls_name for cls_name, i in class_to_idx.items()}

# 时域拉伸类（复数域操作）
class ComplexTimeStretch:
    def __init__(self, sample_rate, p=0.5):
        self.time_stretch = T.TimeStretch()
        self.p = p
        self.sample_rate = sample_rate
        self.n_fft = 2048
        self.hop_length = 300
    
    def __call__(self, waveform):
        if torch.rand(1).item() < self.p:
            # 转换为复数频谱
            spec = torch.stft(
                waveform.squeeze(0), 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                win_length=1200,
                return_complex=True
            )
            
            # 随机时域拉伸
            rate = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
            stretched_spec = self.time_stretch(spec, rate)
            
            # 转回波形
            stretched_waveform = torch.istft(
                stretched_spec, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                win_length=1200,
                length=waveform.shape[1]
            )
            return stretched_waveform.unsqueeze(0)
        return waveform

# 频谱增强类
class SpecAugment:
    def __init__(self, freq_mask_param=15, time_mask_param=35, p=0.5):
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        self.p = p
    
    def __call__(self, spec):
        # 确保输入是3D [channels, freq, time]
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
        
        # 随机频率掩蔽
        if torch.rand(1).item() < self.p:
            spec = self.freq_mask(spec)
        
        # 随机时间掩蔽
        if torch.rand(1).item() < self.p:
            spec = self.time_mask(spec)
        
        # 随机增益 (在分贝域操作)
        if torch.rand(1).item() < self.p:
            gain = torch.FloatTensor(1).uniform_(-6, 6)
            spec = spec + gain
        
        return spec.squeeze(0)  # 移除通道维度，返回2D

# 自定义数据集类
class AudioDataset(Dataset):
    def __init__(self, data_path, transform=None, time_stretch=None, return_waveform=False):
        self.data_path = data_path
        self.transform = transform  # 频谱图增强
        self.time_stretch = time_stretch  # 时域拉伸
        self.return_waveform = return_waveform  # 是否返回原始波形
        self.file_paths = []
        self.labels = []
        self.raw_waveforms = []  # 存储原始波形路径
        
        # 收集所有音频文件路径和标签
        for class_name in class_names:
            class_dir = os.path.join(data_path, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.wav'):
                    self.file_paths.append(os.path.join(class_dir, file))
                    self.labels.append(class_to_idx[class_name])
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # 加载音频
        waveform, orig_sr = torchaudio.load(audio_path)
        
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
        
        # 应用时域拉伸（如果需要）
        if self.time_stretch:
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
        
        # 数据增强（在频谱图上）
        if self.transform:
            mel_specgram = self.transform(mel_specgram)
        
        # 添加通道维度 (1, n_mels, time)
        if mel_specgram.dim() == 2:
            mel_specgram = mel_specgram.unsqueeze(0)
        
        # 返回原始波形用于播放
        if self.return_waveform:
            return mel_specgram, label, waveform, audio_path
        return mel_specgram, label

# 创建数据增强对象
time_stretch = ComplexTimeStretch(SAMPLE_RATE, p=0.5)
spec_augment = SpecAugment(freq_mask_param=15, time_mask_param=35, p=0.5)

# 加载完整数据集（训练集需要原始波形用于后续播放）
full_dataset = AudioDataset(DATA_PATH, transform=spec_augment, time_stretch=time_stretch, return_waveform=True)

# 划分训练集和验证集 (80%训练, 20%验证)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size]
)

# 验证集不需要数据增强
val_dataset.dataset.transform = None
val_dataset.dataset.time_stretch = None

# 数据加载器
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)

# 定义预训练模型
class CNN14(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super(CNN14, self).__init__()
        
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 5
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
        # 确保输入是4D [batch, channels, height, width]
        if x.dim() == 5:
            x = x.squeeze(1)  # 移除多余的维度
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 初始化模型
model = CNN14(num_classes=N_CLASSES).to(device)

# 加载预训练权重
pretrained_dict = torch.load('/kaggle/working/cnn14.pth')
model_dict = model.state_dict()

# 1. 提取模型参数并重命名
pretrained_dict = {k.replace('model.', ''): v for k, v in pretrained_dict.items() 
                  if k.startswith('model') and 'fc' not in k}

# 2. 更新当前模型参数
model_dict.update(pretrained_dict)

# 3. 加载更新后的参数（允许部分不匹配）
model.load_state_dict(model_dict, strict=False)

print("✅ 预训练权重加载成功（忽略全连接层）")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)

# 训练历史记录
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# 训练函数
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data in tqdm(loader, desc='Training'):
        if len(data) == 4:  # 包含波形数据
            inputs, labels, _, _ = data
        else:
            inputs, labels = data
            
        # 确保输入维度正确
        if inputs.dim() == 5:
            inputs = inputs.squeeze(1)  # 从 [B, 1, 1, H, W] -> [B, 1, H, W]
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# 验证函数
def validate(model, loader, criterion, return_predictions=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_waveforms = []
    all_paths = []
    
    with torch.no_grad():
        for data in tqdm(loader, desc='Validating'):
            if len(data) == 4:  # 包含波形数据
                inputs, labels, waveforms, paths = data
                all_waveforms.extend(waveforms)
                all_paths.extend(paths)
            else:
                inputs, labels = data
                
            # 确保输入维度正确
            if inputs.dim() == 5:
                inputs = inputs.squeeze(1)  # 从 [B, 1, 1, H, W] -> [B, 1, H, W]
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    if return_predictions:
        return epoch_loss, epoch_acc, all_labels, all_preds, all_waveforms, all_paths
    return epoch_loss, epoch_acc

# 训练循环
NUM_EPOCHS = 150  # 设置较大的epoch数，由早停机制控制
best_val_acc = 0.0
patience = 15  # 早停等待周期
no_improve_epochs = 0  # 记录没有提升的epoch数
best_model_path = '/kaggle/working/best_model.pth'

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    # 训练阶段
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    
    # 验证阶段
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    # 更新学习率
    scheduler.step(val_acc)
    
    # 记录历史
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        no_improve_epochs = 0  # 重置计数器
        print(f"Saved new best model with val acc: {best_val_acc:.4f}")
    else:
        no_improve_epochs += 1
        print(f"No improvement for {no_improve_epochs}/{patience} epochs")
    
    # 提前停止判断
    if no_improve_epochs >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

# 训练后可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss Evolution')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy Evolution')
plt.legend()
plt.savefig('/kaggle/working/training_history.png')
plt.show()

print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# 加载最佳模型进行最终评估
model.load_state_dict(torch.load(best_model_path))
model.eval()

# 在验证集上生成预测结果
val_loss, val_acc, all_labels, all_preds, all_waveforms, all_paths = validate(
    model, val_loader, criterion, return_predictions=True
)

# 生成混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png')
plt.show()

# 生成分类报告
report = classification_report(all_labels, all_preds, target_names=class_names)
print("Classification Report:")
print(report)

# 保存分类报告为文本文件
with open('/kaggle/working/classification_report.txt', 'w') as f:
    f.write(report)

# 随机选择10个样本进行可视化
sample_indices = random.sample(range(len(all_labels)), min(5, len(all_labels)))
print("\nRandom Samples from Validation Set:")

# 创建结果文件夹
os.makedirs('/kaggle/working/samples', exist_ok=True)

for i, idx in enumerate(sample_indices):
    true_label = idx_to_class[all_labels[idx]]
    pred_label = idx_to_class[all_preds[idx]]
    waveform = all_waveforms[idx]
    audio_path = all_paths[idx]
    
    # 保存音频文件
    audio_filename = f"/kaggle/working/samples/sample_{i}_{true_label}_pred_{pred_label}.wav"
    torchaudio.save(audio_filename, waveform, SAMPLE_RATE)
    
    # 显示样本信息
    print(f"\nSample {i+1}:")
    print(f"Audio Path: {audio_path}")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {pred_label}")
    print(f"Saved as: {audio_filename}")
    
    # 尝试在notebook中播放音频
    try:
        print("Playing audio...")
        display(ipd.Audio(waveform.numpy(), rate=SAMPLE_RATE))
    except:
        print("Audio playback not supported in this environment. Please download the file to listen.")
    
    # 可视化波形和频谱图
    plt.figure(figsize=(15, 5))
    
    # 波形图
    plt.subplot(1, 2, 1)
    plt.plot(waveform[0].numpy().T)
    plt.title(f"Waveform\nTrue: {true_label}, Pred: {pred_label}")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    # 频谱图
    plt.subplot(1, 2, 2)
    spectrogram = T.Spectrogram()(waveform)
    plt.imshow(spectrogram.log2()[0,:,:].numpy(), cmap='viridis', aspect='auto', origin='lower')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f'/kaggle/working/samples/sample_{i}_spectrogram.png')
    plt.show()

print("\nAll results saved to /kaggle/working/ directory")