import base64
import re
import os

def extract_mermaid_diagrams(markdown_file):
    """
    Extract Mermaid diagram code blocks from a markdown file
    """
    if not os.path.exists(markdown_file):
        print(f"File not found: {markdown_file}")
        return []
    
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all mermaid code blocks
    pattern = r'```mermaid\n(.*?)\n```'
    diagrams = re.findall(pattern, content, re.DOTALL)
    
    return diagrams

def render_mermaid_diagram(mermaid_code, title="Diagram"):
    """
    Render a Mermaid diagram using the Mermaid API
    """
    # Clean up the mermaid code
    mermaid_code = mermaid_code.strip()
    
    # Encode mermaid code
    graphbytes = mermaid_code.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    
    # Generate image URL
    img_url = f"https://mermaid.ink/img/{base64_string}"
    
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"🔗 Image URL: {img_url}")
    print(f"📋 Copy this URL and paste into your browser to view the diagram")
    print(f"{'='*50}\n")
    
    return img_url

def main():
    # Extract diagrams from your RAG architecture file
    markdown_file = "RAG_ARCHITECTURE.md"
    diagrams = extract_mermaid_diagrams(markdown_file)
    
    if not diagrams:
        print(f"❌ No Mermaid diagrams found in {markdown_file}")
        return
    
    print(f"🎯 Found {len(diagrams)} Mermaid diagrams in {markdown_file}")
    
    # Generate URLs for each diagram
    urls = []
    titles = [
        "Enterprise RAG System Architecture", 
        "Query Processing & Ranking Architecture"
    ]
    
    for i, diagram in enumerate(diagrams):
        title = titles[i] if i < len(titles) else f"Diagram {i+1}"
        url = render_mermaid_diagram(diagram, title)
        urls.append(url)
    
    print("\n" + "="*60)
    print("🚀 QUICK ACCESS - All Diagram URLs:")
    print("="*60)
    for i, url in enumerate(urls):
        print(f"{i+1}. {url}")
    
    print("\n💡 Instructions:")
    print("1. Copy any URL from above")
    print("2. Paste into your web browser")  
    print("3. View your rendered diagram!")
    print("\n🎨 Alternative: Use Ctrl+Shift+V in VS Code for live preview")

if __name__ == "__main__":
    main()