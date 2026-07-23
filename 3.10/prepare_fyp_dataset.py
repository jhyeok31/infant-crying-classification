import os
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
import soundfile as sf

def prepare_dataset(source_dir="/app/FYP dataset", output_dir="/app/processed_fyp_dataset", test_size=0.2):
    """
    Prepare the FYP dataset by organizing and splitting audio files.
    """
    output_path = Path(output_dir)
    train_path = output_path / "train"
    test_path = output_path / "test"
    
    for path in [train_path, test_path]:
        path.mkdir(parents=True, exist_ok=True)
        
    class_mapping = {
        'belly pain': 'bp',
        'burping': 'bu',
        'discomfort': 'dc',
        'hungry': 'hu',
        'non-crying': 'nc',
        'tired': 'ti'
    }
    
    metadata = []
    
    source_path = Path(source_dir)
    print(f"Reading dataset from: {source_path}")
    
    for class_folder in source_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            if class_name not in class_mapping:
                print(f"Skipping unknown class folder: {class_name}")
                continue
            class_code = class_mapping[class_name]
            
            audio_files = list(class_folder.glob("*.wav")) + \
                         list(class_folder.glob("*.mp3")) + \
                         list(class_folder.glob("*.caf")) + \
                         list(class_folder.glob("*.3gp")) + \
                         list(class_folder.glob("*.ogg"))
            
            if len(audio_files) == 0:
                print(f"No audio files found in {class_folder}")
                continue
                
            print(f"Found {len(audio_files)} files for class: {class_name}")
            
            train_files, test_files = train_test_split(
                audio_files, test_size=test_size, random_state=42
            )
            
            for files, split_path in [(train_files, train_path), (test_files, test_path)]:
                for audio_file in files:
                    try:
                        y, sr = librosa.load(audio_file, sr=16000)
                        
                        new_filename = f"{audio_file.stem}-{class_code}.wav"
                        output_file = split_path / new_filename
                        
                        sf.write(output_file, y, sr, subtype='PCM_16')
                        
                        metadata.append({
                            'filename': new_filename,
                            'class': class_name,
                            'class_code': class_code,
                            'split': 'train' if split_path == train_path else 'test'
                        })
                        
                    except Exception as e:
                        print(f"Error processing {audio_file}: {str(e)}")
    
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_path / 'metadata.csv', index=False)
    
    print(f"Dataset prepared successfully in {output_dir}")
    if not metadata_df.empty:
        print("\nClass distribution:")
        print(metadata_df.groupby(['split', 'class']).size().unstack(fill_value=0))

if __name__ == "__main__":
    prepare_dataset()
