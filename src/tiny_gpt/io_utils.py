# Export function
def io_load_text_file(file_path="../The_Verdict.txt") -> str:
    """
    Load a text file and return its content as a string.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Raw text content of the file.
    """

    print(f"Running {io_load_text_file.__name__}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        print(f"Loaded '{file_path}' successfully!")
        print(f"Total number of characters: {len(raw_text)}")
        print(f"Preview (first 100 chars):\n{raw_text[:100]}")
        return raw_text
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return ""
