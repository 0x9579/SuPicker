import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from supicker.data.star_parser import parse_star_file

def test_star_parsing():
    star_path = Path("tests/data/fixtures/test.star")
    if not star_path.exists():
        print(f"File not found: {star_path}")
        return
        
    try:
        result = parse_star_file(star_path)
        print(f"Successfully parsed {len(result)} micrographs from {star_path}")
        for mic, particles in result.items():
            print(f"  {mic}: {len(particles)} particles")
    except Exception as e:
        print(f"Error parsing STAR file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_star_parsing()
