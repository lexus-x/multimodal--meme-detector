"""Data augmentation and preprocessing utilities."""
import random
import torch
from PIL import Image, ImageFilter, ImageEnhance


class TextAugmentor:
    """Simple text augmentation for memes."""
    
    def __init__(self, p: float = 0.3):
        self.p = p
    
    def __call__(self, text: str) -> str:
        if random.random() > self.p:
            return text
        
        augmentations = [
            self._swap_words,
            self._drop_word,
            self._duplicate_word,
        ]
        return random.choice(augmentations)(text)
    
    def _swap_words(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
        return " ".join(words)
    
    def _drop_word(self, text: str) -> str:
        words = text.split()
        if len(words) < 3:
            return text
        words.pop(random.randint(0, len(words) - 1))
        return " ".join(words)
    
    def _duplicate_word(self, text: str) -> str:
        words = text.split()
        if not words:
            return text
        word = random.choice(words)
        idx = random.randint(0, len(words))
        words.insert(idx, word)
        return " ".join(words)


class ImageAugmentor:
    """Additional image augmentation strategies."""
    
    def __init__(self, p: float = 0.3):
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        
        augmentations = [
            lambda im: im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5))),
            lambda im: ImageEnhance.Brightness(im).enhance(random.uniform(0.8, 1.2)),
            lambda im: ImageEnhance.Contrast(im).enhance(random.uniform(0.8, 1.2)),
            lambda im: im.rotate(random.uniform(-10, 10), fillcolor=(128, 128, 128)),
        ]
        return random.choice(augmentations)(img)
