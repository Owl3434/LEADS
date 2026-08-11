"""
# ============================================================================
# IA3: Parameter-Efficient Fine-Tuning
# ============================================================================
#
# This implementation is based on the (IA)^3 method introduced in:
#
# H. Liu, D. Tam, M. Muqeeth, J. Mohta, T. Huang, M. Bansal,
# and C. Raffel,
# "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper
# than In-Context Learning,"
# Advances in Neural Information Processing Systems (NeurIPS), 2022.
#
# arXiv: 2205.05638
#
# Official implementation:
#   https://github.com/r-three/t-few
#
# NOTE:
# This code implements the IA3 adaptation mechanism directly.
# The pretrained model parameters are frozen, while learned vectors
# multiplicatively rescale intermediate activations.
#
# In this implementation, trainable IA3 scaling vectors are applied to
# the k_proj, v_proj, and up_proj outputs:
#
#     h' = h * l
#
# where l is a learned IA3 scaling vector.
#
# ============================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from PIL import Image
import json
import math
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# IA3-Llama 模型
# ============================================================================

class IA3LlamaModel(nn.Module):
    """IA3-Llama模型"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__()
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": self.device},
            trust_remote_code=True
        )
        
        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_attention_heads)
        self.head_dim = self.hidden_dim // self.num_attention_heads
        self.kv_dim = self.head_dim * self.num_key_value_heads
        
        # 冻结基础模型
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 初始化IA3参数
        self.ia3_parameters = nn.ParameterDict()
        for layer_idx in range(self.num_layers):
            for module_name in ["k_proj", "v_proj", "up_proj"]:
                param_name = f"layer_{layer_idx}_{module_name}"
                
                if module_name in ["k_proj", "v_proj"]:
                    dim = self.kv_dim
                elif module_name == "up_proj":
                    dim = self.intermediate_size
                
                param = nn.Parameter(torch.ones(dim, device=self.device, dtype=torch.float32))
                self.ia3_parameters[param_name] = param
        
        # 注册钩子
        self.hooks = []
        for layer_idx, layer in enumerate(self.model.model.layers):
            for module_name in ["k_proj", "v_proj", "up_proj"]:
                param_name = f"layer_{layer_idx}_{module_name}"
                
                if module_name == "k_proj":
                    target_module = layer.self_attn.k_proj
                elif module_name == "v_proj":
                    target_module = layer.self_attn.v_proj
                elif module_name == "up_proj":
                    target_module = layer.mlp.up_proj
                
                ia3_param = self.ia3_parameters[param_name]
                
                def make_hook(param):
                    def hook(module, input, output):
                        # 将IA3参数转换到output的精度
                        if param.dtype != output.dtype:
                            param_converted = param.to(output.dtype)
                        else:
                            param_converted = param
                        return output * param_converted
                    return hook
                
                handle = target_module.register_forward_hook(make_hook(ia3_param))
                self.hooks.append(handle)
    
    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict
        )
    
    def get_ia3_parameters(self):
        """获取IA3参数"""
        return list(self.ia3_parameters.values())
    
    def save_ia3_weights(self, save_path: str):
        """保存IA3权重为FP16格式"""
        save_path = Path(save_path)
        base_path = save_path.with_suffix('')
        
        # 保存二进制权重（FP16）
        bin_path = f"{base_path}.bin"
        all_data = []
        param_order = []
        param_info = {}
        
        for layer_idx in range(self.num_layers):
            for module_name in ['k_proj', 'v_proj', 'up_proj']:
                param_name = f"layer_{layer_idx}_{module_name}"
                if param_name in self.ia3_parameters:
                    param = self.ia3_parameters[param_name]
                    
                    # 转换为FP16并保存
                    fp16_data = param.detach().cpu().half().numpy()
                    
                    all_data.append(fp16_data.flatten())
                    param_order.append(param_name)
                    param_info[param_name] = {
                        'shape': list(param.shape),
                        'size': param.numel()
                    }
        
        # 保存二进制数据
        with open(bin_path, 'wb') as f:
            for data in all_data:
                f.write(data.tobytes())
        
        # 计算总大小 (FP16: 2 bytes per element)
        total_bytes = sum(info['size'] * 2 for info in param_info.values())
        
        # 保存元数据
        metadata = {
            'quantization': 'fp16',
            'bytes_per_element': 2,
            'parameter_order': param_order,
            'parameter_info': param_info,
            'total_bytes': total_bytes,
            'ia3_config': {
                'num_layers': self.num_layers,
                'kv_dim': self.kv_dim,
                'intermediate_size': self.intermediate_size
            },
            'model_config': {
                'model_path': self.model_path
            }
        }
        
        metadata_path = f"{base_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ IA3 weights saved (FP16):")
        print(f"  Binary: {bin_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Total parameters: {sum(info['size'] for info in param_info.values()):,}")
        print(f"  File size: {total_bytes:,} bytes ({total_bytes/1024:.2f} KB)")
    
    def load_ia3_weights(self, load_path: str, use_fp16: bool = True):
        """
        加载IA3权重（FP16格式）
        
        Args:
            load_path: 权重文件路径
            use_fp16: 是否使用FP16精度（推理时建议True，训练时False）
        """
        load_path = Path(load_path)
        base_path = load_path.with_suffix('')
        
        # 加载元数据
        metadata_path = f"{base_path}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        param_order = metadata['parameter_order']
        param_info = metadata['parameter_info']
        
        # 加载二进制数据
        bin_path = f"{base_path}.bin"
        with open(bin_path, 'rb') as f:
            binary_data = f.read()
        
        # 解析权重
        offset = 0
        for param_name in param_order:
            if param_name not in self.ia3_parameters:
                continue
            
            info = param_info[param_name]
            shape = info['shape']
            size = info['size']
            
            # 读取FP16数据 (2 bytes per element)
            read_bytes = size * 2
            chunk = binary_data[offset:offset + read_bytes]
            fp16_data = np.frombuffer(chunk, dtype=np.float16)
            offset += read_bytes
            
            # 重塑形状
            fp16_data = fp16_data.reshape(shape)
            
            # 转换精度
            if use_fp16:
                target_dtype = torch.float16
            else:
                target_dtype = torch.float32
            
            param_tensor = torch.from_numpy(fp16_data.astype(np.float32)).to(
                device=self.device,
                dtype=target_dtype
            )
            
            # 加载到模型（直接替换data以改变dtype）
            with torch.no_grad():
                self.ia3_parameters[param_name].data = param_tensor
        
        print(f"✓ Loaded FP16 weights from {bin_path}")


# ============================================================================
# 数据加载器（添加反向词表）
# ============================================================================

class ImageDataLoader:
    """图像数据加载器（修正版）"""
    
    def __init__(self, image_path: str, model_path: str, prompt_type: str = "instruction", 
                 patch_size: int = 16, batch_size: int = 8):
        self.image_path = image_path
        self.model_path = model_path
        self.prompt_type = prompt_type
        self.patch_size = patch_size
        self.batch_size = batch_size
        
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.uint8)
        
        h, w = img_array.shape[:2]
        new_h = (h // patch_size) * patch_size
        new_w = (w // patch_size) * patch_size
        self.image = img_array[:new_h, :new_w]
        self.image_height, self.image_width = self.image.shape[:2]
        
        # Patch信息
        self.patches_per_row = self.image_width // patch_size
        self.patches_per_col = self.image_height // patch_size
        self.total_patches = self.patches_per_row * self.patches_per_col
        
        # 初始化tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token_id = 128000
        
        # 构建数字字典（双向映射）
        self.digital_dict = {}  # 像素值 -> token_id
        self.reverse_digital_dict = {}  # token_id -> 像素值
        
        for i in range(256):
            tokens = self.tokenizer.encode(str(i), add_special_tokens=False)
            if len(tokens) != 1:
                raise ValueError(f"Number {i} maps to {len(tokens)} tokens")
            token_id = tokens[0]
            self.digital_dict[i] = token_id
            self.reverse_digital_dict[token_id] = i
        
        print(f"✓ Image loaded: {self.image_width}x{self.image_height}")
        print(f"✓ Total patches: {self.total_patches}")
        print(f"✓ Digital dictionary: 256 pixel values mapped to tokens")
    
    def get_patch_at_pixel(self, x: int, y: int) -> np.ndarray:
        """获取patch"""
        patch_row = y // self.patch_size
        patch_col = x // self.patch_size
        start_y = patch_row * self.patch_size
        start_x = patch_col * self.patch_size
        return self.image[start_y:start_y+self.patch_size, 
                         start_x:start_x+self.patch_size]
    
    def tokenize_patch(self, patch: np.ndarray, x: int, y: int) -> Dict:
        """将patch转换为tokens"""
        # 提示词
        if self.prompt_type == "no_prompt":
            prompt_tokens = [self.tokenizer.bos_token_id]
        elif self.prompt_type == "channel":
            prompt = "Every triplet denote an RGB pixel of a 2D image. Predict the next RGB pixel based on the previous pixels."
            prompt_tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt, add_special_tokens=False)
        else:  # instruction
            prompt = f"At position ({x:04d}, {y:04d}) begins a sequence where every triplet denotes an RGB pixel of a 2D image. Predict the next RGB pixel based on the previous pixels."
            prompt_tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt, add_special_tokens=False)
        
        prompt_length = len(prompt_tokens)
        
        # 像素tokens
        patch_flat = patch.flatten()
        pixel_tokens = [self.digital_dict[int(val)] for val in patch_flat]
        
        # 组合
        all_tokens = prompt_tokens + pixel_tokens
        attention_mask = [1] * len(all_tokens)
        labels = all_tokens.copy()
        for i in range(prompt_length):
            labels[i] = -100
        
        return {
            'input_ids': torch.tensor(all_tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'prompt_length': prompt_length
        }
    
    def get_all_patches(self):
        """获取所有patches"""
        all_data = []
        for row in range(self.patches_per_col):
            for col in range(self.patches_per_row):
                x = col * self.patch_size
                y = row * self.patch_size
                patch = self.get_patch_at_pixel(x, y)
                tokens_dict = self.tokenize_patch(patch, x, y)
                all_data.append(tokens_dict)
        
        return {
            'input_ids': torch.stack([d['input_ids'] for d in all_data]),
            'attention_mask': torch.stack([d['attention_mask'] for d in all_data]),
            'labels': torch.stack([d['labels'] for d in all_data])
        }


# ============================================================================
# 训练器
# ============================================================================

class Trainer:
    """训练器 - 使用标准交叉熵"""
    
    def __init__(self, model: IA3LlamaModel, dataloader: ImageDataLoader, 
                 output_dir: str, learning_rate: float = 1e-3, epochs: int = 10):
        self.model = model
        self.dataloader = dataloader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.device = model.device
        
        # 优化器
        self.optimizer = optim.AdamW(model.get_ia3_parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # 混合精度
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
        
        self.best_loss = float('inf')
    
    def train(self):
        """训练"""
        print(f"\n{'='*70}")
        print(f"Starting Training (Cross-Entropy Loss)")
        print(f"  Epochs: {self.epochs}")
        print(f"  Device: {self.device}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"{'='*70}\n")
        
        # 获取所有数据
        all_data = self.dataloader.get_all_patches()
        input_ids = all_data['input_ids']
        attention_mask = all_data['attention_mask']
        labels = all_data['labels']
        total_patches = len(input_ids)
        batch_size = self.dataloader.batch_size
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0
            num_batches = (total_patches + batch_size - 1) // batch_size
            
            with tqdm(total=num_batches, desc=f"Epoch {epoch}/{self.epochs}") as pbar:
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, total_patches)
                    
                    batch_input_ids = input_ids[start_idx:end_idx].to(self.device)
                    batch_attention_mask = attention_mask[start_idx:end_idx].to(self.device)
                    batch_labels = labels[start_idx:end_idx].to(self.device)
                    
                    if self.use_amp:
                        with autocast():
                            outputs = self.model(batch_input_ids, batch_attention_mask, 
                                               labels=batch_labels, return_dict=True)
                            loss = outputs.loss
                        
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.get_ia3_parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        outputs = self.model(batch_input_ids, batch_attention_mask, 
                                           labels=batch_labels, return_dict=True)
                        loss = outputs.loss
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.get_ia3_parameters(), 1.0)
                        self.optimizer.step()
                    
                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    
                    pbar.set_postfix({
                        'loss': f'{batch_loss:.4f}'
                    })
                    pbar.update(1)
            
            avg_loss = epoch_loss / num_batches
            self.scheduler.step()
            
            print(f"Epoch {epoch}/{self.epochs} - Loss: {avg_loss:.4f}")
            
            # 保存最佳模型
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                save_path = self.output_dir / "best_model.bin"
                self.model.save_ia3_weights(save_path)
                print(f"✓ Best model saved (loss: {avg_loss:.4f})")
        
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"{'='*70}\n")


# ============================================================================
# 测试器（在256维像素空间上计算理论值）
# ============================================================================

class Tester:
    """图像测试器（修正版）"""
    
    def __init__(self, model: IA3LlamaModel, dataloader: ImageDataLoader, 
                 output_dir: str, weights_path: Optional[str] = None):
        self.model = model
        self.dataloader = dataloader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = model.device
        self.weights_path = weights_path
    
    def _quantize_probabilities_vectorized(self, pdf_batch, precision=32, min_freq=1):
        """
        快速频率量化（完全矢量化，无循环）
        
        Args:
            pdf_batch: shape [batch_size, 256] 的概率分布
            precision: 算术编码精度
            min_freq: 最小频率
            
        Returns:
            quantized_freqs: shape [batch_size, 256] 的量化后频率
        """
        batch_size, n_symbols = pdf_batch.shape
        width = (2 ** precision - 1) + 1
        
        # 归一化概率
        pdf_normalized = pdf_batch / (pdf_batch.sum(dim=1, keepdim=True) + 1e-10)
        
        # 预留最小频率空间
        reserved = n_symbols * min_freq
        available = width - reserved
        
        # 使用round而不是floor，减少舍入误差
        freqs = torch.round(pdf_normalized * available).long() + min_freq
        
        # 一次性调整到精确值（完全矢量化）
        diff = width - freqs.sum(dim=1)
        max_indices = torch.argmax(pdf_batch, dim=1)
        freqs[torch.arange(batch_size), max_indices] += diff
        
        return freqs
    
    def test(self):
        """测试图像（修正版 - 256维像素空间 + 算术编码频率量化）"""
        print(f"\n{'='*70}")
        print(f"Starting Test (256-dim + Arithmetic Coding Quantization)")
        print(f"{'='*70}\n")
        
        self.model.eval()
        start_time = time.time()
        
        # 预先构建像素token索引（加速）
        pixel_token_indices = torch.tensor(
            [self.dataloader.digital_dict[i] for i in range(256)],
            dtype=torch.long,
            device=self.device
        )
        
        total_bits = 0
        positions = []
        for row in range(self.dataloader.patches_per_col):
            for col in range(self.dataloader.patches_per_row):
                x = col * self.dataloader.patch_size
                y = row * self.dataloader.patch_size
                positions.append((x, y))
        
        PRECISION = 32
        WIDTH = (2 ** PRECISION - 1) + 1
        
        with torch.no_grad():
            for x, y in tqdm(positions, desc="Testing"):
                patch = self.dataloader.get_patch_at_pixel(x, y)
                tokens_dict = self.dataloader.tokenize_patch(patch, x, y)
                
                input_ids = tokens_dict['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = tokens_dict['attention_mask'].unsqueeze(0).to(self.device)
                prompt_length = tokens_dict['prompt_length']
                
                outputs = self.model(input_ids, attention_mask)
                logits = outputs.logits[0]
                
                # 批量收集所有像素位置的概率分布
                batch_probs = []
                target_pixels = []
                
                for pos in range(prompt_length, len(input_ids[0])):
                    target_token = input_ids[0, pos].item()
                    logit_pos = pos - 1
                    
                    if logit_pos >= len(logits):
                        break
                    
                    # 提取256个像素token的logits
                    pixel_logits = logits[logit_pos, pixel_token_indices]
                    
                    # 在256维空间上计算softmax概率
                    pixel_probs = F.softmax(pixel_logits, dim=0)
                    
                    # 归一化（模拟算术编码器的_normalize_pdf）
                    machine_epsilon = torch.finfo(pixel_probs.dtype).eps
                    pdf_sum = pixel_probs.sum()
                    pixel_probs = pixel_probs / pdf_sum
                    pixel_probs = (1 - 2 * 256 * machine_epsilon) * pixel_probs + machine_epsilon
                    
                    batch_probs.append(pixel_probs)
                    target_pixel_val = self.dataloader.reverse_digital_dict[target_token]
                    target_pixels.append(target_pixel_val)
                
                if len(batch_probs) == 0:
                    continue
                
                # 批量量化频率
                batch_probs = torch.stack(batch_probs)  # shape: [768, 256]
                quantized_freqs = self._quantize_probabilities_vectorized(batch_probs, PRECISION, min_freq=1)
                
                # 计算理论比特数：-log2(freq / WIDTH)
                patch_bits = 0.0
                for i, target_pixel in enumerate(target_pixels):
                    freq = quantized_freqs[i, target_pixel].item()
                    # 理论比特数 = -log2(freq / WIDTH)
                    bits = -math.log2(freq / WIDTH)
                    patch_bits += bits
                
                total_bits += patch_bits
        
        encoding_time = time.time() - start_time
        
        # 计算统计
        original_bytes = self.dataloader.image_width * self.dataloader.image_height * 3
        compressed_bytes = total_bits / 8
        
        # IA3权重大小 (FP16: 2 bytes per element)
        if self.weights_path and Path(self.weights_path).exists():
            # 读取实际文件大小
            ia3_bytes = Path(self.weights_path).stat().st_size
            weights_source = "actual file size"
        else:
            # 估算（FP16: 2 bytes per element）
            ia3_params = sum(p.numel() for p in self.model.get_ia3_parameters())
            ia3_bytes = ia3_params * 2  # fp16
            weights_source = "estimated (fp16)"
        
        total_compressed_bytes = compressed_bytes + ia3_bytes
        compression_ratio = original_bytes / total_compressed_bytes
        bpp = (total_compressed_bytes * 8) / (self.dataloader.image_width * self.dataloader.image_height)
        bps = bpp / 3
        
        # 打印结果
        print(f"\n{'='*70}")
        print(f"Test Results")
        print(f"{'='*70}")
        print(f"Original size:        {original_bytes:>10,} bytes")
        print(f"Compressed data:      {compressed_bytes:>10,.0f} bytes")
        print(f"IA3 weights:          {ia3_bytes:>10,} bytes ({ia3_bytes/1024:.2f} KB) [{weights_source}]")
        print(f"Total compressed:     {total_compressed_bytes:>10,.0f} bytes")
        print(f"Compression ratio:    {compression_ratio:>10.2f}x")
        print(f"Bits per pixel:       {bpp:>10.3f}")
        print(f"Bits per subpixel:    {bps:>10.3f}")
        print(f"Encoding time:        {encoding_time:>10.2f} seconds")
        print(f"{'='*70}\n")
        
        # 保存结果
        results = {
            'original_bytes': original_bytes,
            'compressed_bytes': float(compressed_bytes),
            'ia3_bytes': ia3_bytes,
            'total_compressed_bytes': float(total_compressed_bytes),
            'compression_ratio': float(compression_ratio),
            'bits_per_pixel': float(bpp),
            'bits_per_subpixel': float(bps),
            'encoding_time': encoding_time
        }
        
        result_file = self.output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {result_file}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='IA3 Image Compression - Train and Test (FP16 + CE)')
    
    # 模式选择
    parser.add_argument('mode', choices=['train', 'test', 'train_and_test'], 
                       help='Mode: train, test, or train_and_test')
    
    # 通用参数
    parser.add_argument('-i', '--image', type=str, required=True, help='Input image path')
    parser.add_argument('-m', '--model', type=str, required=True, help='Llama model path (absolute)')
    parser.add_argument('-o', '--output', type=str, default='./output', help='Output directory')
    parser.add_argument('-p', '--prompt_type', type=str, default='instruction',
                       choices=['no_prompt', 'channel', 'instruction'], help='Prompt type')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    # 测试参数
    parser.add_argument('-w', '--weights', type=str, help='IA3 weights path (for test mode)')
    
    args = parser.parse_args()
    
    # 验证路径
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    print(f"\n{'='*70}")
    print(f"IA3 Image Compression (FP16 + Cross-Entropy)")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt_type}")
    print(f"Storage format: FP16 (2 bytes per element)")
    print(f"{'='*70}\n")
    
    # 初始化
    model = IA3LlamaModel(args.model, args.device)
    dataloader = ImageDataLoader(args.image, args.model, args.prompt_type, 
                                 args.patch_size, args.batch_size)
    
    if args.mode == 'train':
        # 训练模式
        trainer = Trainer(model, dataloader, args.output, args.lr, args.epochs)
        trainer.train()
        
    elif args.mode == 'test':
        # 测试模式
        if not args.weights:
            raise ValueError("Please specify --weights for test mode")
        if not Path(args.weights).exists():
            raise FileNotFoundError(f"Weights not found: {args.weights}")
        
        # 加载权重（测试时使用FP16）
        model.load_ia3_weights(args.weights, use_fp16=True)
        
        # 测试
        tester = Tester(model, dataloader, args.output, weights_path=args.weights)
        tester.test()
        
    else:  # train_and_test
        # 训练和测试模式
        print(f"\n{'='*70}")
        print(f"Sequential Execution: Train → Test")
        print(f"{'='*70}\n")
        
        # Step 1: 训练
        trainer = Trainer(model, dataloader, args.output, args.lr, args.epochs)
        trainer.train()
        
        # Step 2: 加载最佳模型权重
        best_weights = Path(args.output) / "best_model.bin"
        if not best_weights.exists():
            raise FileNotFoundError(f"Best model weights not found: {best_weights}")
        
        print(f"\n{'='*70}")
        print(f"Loading best model for testing...")
        print(f"{'='*70}\n")
        model.load_ia3_weights(best_weights, use_fp16=True)
        
        # Step 3: 测试
        tester = Tester(model, dataloader, args.output, weights_path=str(best_weights))
        tester.test()
    
    print("✓ Done!")


if __name__ == "__main__":
    main()
