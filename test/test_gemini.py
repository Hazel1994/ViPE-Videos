#!/usr/bin/env python3
"""
Test script for Gemini API functionality
"""

import os
import json
import datetime
import sys

# Add the helpers directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'helpers'))

from gemini_api import setup_gemini, call_gemini, load_prompt_template, generate_characters, validate_and_save_gemini_response

# Define output directory for all test files
BASE_DIR = "/graphics/scratch2/students/webereli/playground/gemini_test/"
OUTPUT_DIR = f"{BASE_DIR}output"
PROMPT_FILE_PATH = f"{BASE_DIR}prompts/single_additional_prompt.txt"
LYRIC2PROMPT = f"{BASE_DIR}apt_ctx_1_sample_True_vipe_True_abst_0.7_lyric2prompt"
VARIATION = "additional_information2"
ADDITIONAL_INFORAMTION = f"{BASE_DIR}additional_information.txt"

ADDITIONAL_INFO_PREFIX = "Are the prompts fitting with the meaning of the song? If not, fix missinterpretations or ambiguity. Don't explain why. Heres some additional information about the song that you should use for you decision. Only use the information I provide."


def load_prompt_template_with_additional_info(template_path, additional_info_path=None):
    """
    Load prompt template and optionally include additional information
    
    Args:
        template_path (str): Path to the prompt template file
        additional_info_path (str): Path to additional information file (optional)
    
    Returns:
        str: The loaded template with additional info if provided
    """
    try:
        # Load the base template
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        
        # Check if additional information should be included
        additional_info_content = ""
        if additional_info_path and os.path.exists(additional_info_path):
            with open(additional_info_path, 'r', encoding='utf-8') as f:
                additional_info_content = f.read().strip()
        
        # If additional information exists and is not empty, add it with prefix
        if additional_info_content:
            additional_info_with_prefix = f"{ADDITIONAL_INFO_PREFIX}\n\n{additional_info_content}"
        else:
            additional_info_with_prefix = ""
        
        # Replace the placeholder in the template
        template = template.replace("{additional_information_with_prefix}", additional_info_with_prefix)
        
        return template
        
    except FileNotFoundError as e:
        print(f"Template file not found: {e}")
        return None
    except Exception as e:
        print(f"Error loading template: {e}")
        return None

def test_real_character_generation():
    """Test character generation with real lyric2prompt data"""
    print("\n=== REAL CHARACTER GENERATION TEST ===")
    
    try:
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Check if lyric2prompt file exists
        if not os.path.exists(LYRIC2PROMPT):
            print(f"Error: lyric2prompt file not found: {LYRIC2PROMPT}")
            return False
        
        # Load the lyric2prompt data
        with open(LYRIC2PROMPT, 'r') as f:
            lyric2prompt_data = json.load(f)
        
        print(f"Loaded lyric2prompt data with {len(lyric2prompt_data)} entries")
        
        # Load the prompt template with additional information
        template = load_prompt_template_with_additional_info(PROMPT_FILE_PATH, ADDITIONAL_INFORAMTION)
        if not template:
            print("Error: Failed to load prompt template")
            return False
        
        print("Prompt template loaded successfully")
        
        # Check if additional information was included
        if os.path.exists(ADDITIONAL_INFORAMTION):
            with open(ADDITIONAL_INFORAMTION, 'r') as f:
                additional_content = f.read().strip()
            if additional_content:
                print("Additional information included in prompt")
            else:
                print("Additional information file is empty")
        else:
            print("No additional information file found")
        
        # Format the template with the lyric2prompt data
        json_content = json.dumps(lyric2prompt_data, indent=2)
        formatted_prompt = template.format(json_content=json_content)
        
        print(f"Formatted prompt length: {len(formatted_prompt)} characters")
        
        # Save the formatted prompt for inspection
        formatted_prompt_file = os.path.join(OUTPUT_DIR, f"{VARIATION}_formatted_prompt.txt")
        with open(formatted_prompt_file, 'w') as f:
            f.write(formatted_prompt)
        print(f"Formatted prompt saved to: {formatted_prompt_file}")
        
        # Initialize Gemini
        model = setup_gemini()
        print("Gemini model initialized")
        
        # Send the formatted prompt to Gemini
        print("Sending prompt to Gemini...")
        response = call_gemini(model, formatted_prompt)
        
        if response:
            print("Received response from Gemini")
            print(f"Response length: {len(response)} characters")
            
            # Save the raw response
            response_file = os.path.join(OUTPUT_DIR, f"{VARIATION}_gemini_response.txt")
            with open(response_file, 'w') as f:
                f.write(response)
            print(f"Raw response saved to: {response_file}")
            
            # Try to validate and save the response using the existing function
            try:
                success, updated_prompts_path, characters_path = validate_and_save_gemini_response(
                    gemini_response=response,
                    original_json_path=LYRIC2PROMPT,
                    output_dir=OUTPUT_DIR
                )
                
                if success:
                    print("Response validation and saving successful!")
                    print(f"Updated prompts: {updated_prompts_path}")
                    print(f"Characters: {characters_path}")
                    
                    # Show some results
                    if characters_path and os.path.exists(characters_path):
                        with open(characters_path, 'r') as f:
                            characters = json.load(f)
                        print(f"Generated {len(characters)} characters:")
                        for char in characters:
                            print(f"  - {char.get('name', 'Unknown')}: {char.get('description', 'No description')}")
                    
                    return True
                else:
                    print("Response validation failed")
                    return False
                    
            except Exception as e:
                print(f"Error during response validation: {e}")
                print("But the API call itself was successful")
                return True  # Consider it a partial success
        else:
            print("No response received from Gemini")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":

    test_real_character_generation()