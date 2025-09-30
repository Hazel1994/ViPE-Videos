import json
import re
import os

class Character:
    """A character object that tracks character information and in which lines they appear"""
    
    def __init__(self, name, description, line_occurrences=None, regularization_images=None, training_images=None):
        """
        Initialize a Character object
        
        Args:
            name (str): Name of the character
            description (str): Description of the character
            line_occurrences (list): List of integers representing line numbers where character appears
            regularization_images (str): Path to the regularization images
            training_images (str): Path to the training images
        """
        self.name = name
        self.description = description
        self.line_occurrences = line_occurrences if line_occurrences is not None else []
        self.regularization_images = regularization_images
        self.training_images = training_images
        self.model_path = None
        
        # Generate unique identifier for DreamBooth training with description
        sanitized_name = self._sanitize_name_for_identifier(name)
        first_descriptor = description.split(',')[0].strip() if description else ""
        self.unique_identifier = f"sks{sanitized_name} {first_descriptor}"
        
        # Keep the bracketed format for Gemini detection
        self.bracketed_name = f"<{name}>"
    
    def _sanitize_name_for_identifier(self, name):
        """
        Sanitize character name for use as unique identifier
        Removes all non-alphanumeric characters to prevent issues
        """
        # Convert to lowercase and keep only letters and numbers
        sanitized = re.sub(r'[^a-z0-9]', '', name.lower())
        return sanitized
    
    def add_occurrence(self, line_number):
        """Add a line number where this character appears"""
        if line_number not in self.line_occurrences:
            self.line_occurrences.append(line_number)
            self.line_occurrences.sort()  # Sort occurrences
    
    def appears_in_line(self, line_number):
        """Check if character appears in a specific line"""
        return line_number in self.line_occurrences
    
    @staticmethod
    def replace_character_names_in_prompts(lyric2prompt, characters):
        """
        Replace character names in prompts with their unique identifiers.
        Transform from <CharacterName> format and plain names to unique identifier format.

        Args:
            lyric2prompt (list): List of prompt dictionaries with 'start', 'end', 'prompt' keys
            characters (list): List of Character objects

        Returns:
            tuple: (animation_prompts dict, updated_lyric2prompt list)
        """
        animation_prompts = {}
        updated_lyric2prompt = []

        fps_p = 15  # frames per second - should match the value from generate_video.py

        for num, l2p in enumerate(lyric2prompt):
            start_frame = int(l2p['start'] * fps_p)

            # Start with the original prompt
            prompt_text = l2p['prompt']

            # Replace character names with unique identifiers
            if characters:
                for character in characters:
                    # Replace bracketed format <CharacterName>
                    bracketed_pattern = f"<{character.name}>"
                    if bracketed_pattern in prompt_text:
                        prompt_text = prompt_text.replace(bracketed_pattern, character.unique_identifier)

                    # Replace plain name with word boundaries (case-insensitive)
                    # Avoid replacing if already replaced (to prevent double replacement)
                    pattern = rf'\b{re.escape(character.name)}\b'
                    # Only replace if the unique_identifier is not already present
                    if re.search(pattern, prompt_text, flags=re.IGNORECASE) and character.unique_identifier not in prompt_text:
                        prompt_text = re.sub(pattern, character.unique_identifier, prompt_text, flags=re.IGNORECASE)

            # Create the animation prompt entry with just the modified prompt text
            animation_prompts[start_frame] = prompt_text

            # Update the lyric2prompt entry
            updated_l2p = l2p.copy()
            updated_l2p['prompt'] = prompt_text
            updated_lyric2prompt.append(updated_l2p)

        return animation_prompts, updated_lyric2prompt

    def to_dict(self):
        """Return a dictionary representation of the Character object."""
        return {
            'name': self.name,
            'description': self.description,
            'line_occurrences': self.line_occurrences,
            'regularization_images': self.regularization_images,
            'training_images': self.training_images
        }
    
    def set_training_images_path(self, path):
        """Set the path to training images for this character"""
        self.training_images = path
    
    def set_regularization_images_path(self, path):
        """Set the path to regularization images for this character"""
        self.regularization_images = path
    
    def get_training_images_path(self):
        """Get the path to training images for this character"""
        return self.training_images
    
    def get_regularization_images_path(self):
        """Get the path to regularization images for this character"""
        return self.regularization_images
    
    def has_trained_model(self):
        """Check if this character has a trained model"""
        return self.model_path and os.path.exists(self.model_path)

    def get_model_path(self):
        """Get the path to the trained model"""
        return getattr(self, 'model_path', None)


def load_characters_from_json(json_file_path):
    """
    Load characters from JSON file and convert to Character objects
    
    Args:
        json_file_path (str): Path to the characters JSON file
        
    Returns:
        list: List of Character objects
    """
    try:
        with open(json_file_path, 'r') as f:
            characters_data = json.load(f)
        
        characters = []
        for char_data in characters_data:
            # Extract data from JSON
            name = char_data.get('name', '')
            description = char_data.get('description', '')
            line_occurrences = char_data.get('line_occurrences', [])
            regularization_images = char_data.get('regularization_images', None)
            training_images = char_data.get('training_images', None)
            
            # Create Character object with name
            character = Character(
                name=name,
                description=description,
                line_occurrences=line_occurrences,
                regularization_images=regularization_images,
                training_images=training_images
            )
            
            characters.append(character)
        
        return characters
        
    except FileNotFoundError:
        print(f"Characters file not found: {json_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing characters JSON: {e}")
        return []
    except Exception as e:
        print(f"Error loading characters: {e}")
        return []

def save_characters_to_json(characters, json_file_path):
    """
    Save Character objects to JSON file
    
    Args:
        characters (list): List of Character objects
        json_file_path (str): Path where to save the JSON file
    """
    try:
        characters_data = []
        for char in characters:
            char_dict = char.to_dict()
            characters_data.append(char_dict)
        
        with open(json_file_path, 'w') as f:
            json.dump(characters_data, f, indent=2)
        
    except Exception as e:
        print(f"Error saving characters: {e}")

def update_character_occurrences(characters, lyric2prompt):
    """
    Scan prompts for character names in brackets and update their line occurrences
    
    Args:
        characters (list): List of Character objects
        lyric2prompt (list): List of dictionaries with 'text' and 'prompt' keys
        
    Returns:
        list: Updated list of Character objects with line occurrences
    """
    
    for line_index, line_data in enumerate(lyric2prompt):
        prompt = line_data.get('prompt', '')
        
        for character in characters:
            # Check for character name in angle brackets <CharacterName> or unique_identifier
            if character.bracketed_name in prompt or character.unique_identifier in prompt:
                character.add_occurrence(line_index)
    
    return characters