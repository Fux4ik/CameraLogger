import os
import shutil
import random

# Create Structure
for split in ['train', 'test']:
    for class_name in ['person', 'no_person']:
        os.makedirs(f'dataset/{split}/{class_name}', exist_ok=True)

# Separate every class
for class_name in ['person', 'no_person']:
    images = [f for f in os.listdir(f'dataset/{class_name}') if f.endswith('.jpg')]
    random.shuffle(images)

    split_idx = int(0.8 * len(images))

    # Train
    for img in images[:split_idx]:
        shutil.move(f'dataset/{class_name}/{img}',
                    f'dataset/train/{class_name}/{img}')

    # Test
    for img in images[split_idx:]:
        shutil.move(f'dataset/{class_name}/{img}',
                    f'dataset/test/{class_name}/{img}')

    print(f'{class_name}: train={split_idx}, test={len(images) - split_idx}')

