from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
from fpdf import FPDF

search = DuckDuckGoSearchRun()

search_tool = Tool(
    name="Search",
    func=search.run,
    description="Search the web using DuckDuckGo"
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = Tool(
    name="Wikipedia",
    func=api_wrapper.run,
    description="Useful for when you need to look up a topic on Wikipedia to get a summary of it. The input to this tool should be a search query."
)

def save_to_pdf(content: str, filename: str = "researcher_output.pdf") -> str:
    """Saves the given content to a PDF file with a timestamped filename if none is provided."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    formatted_text = f"researcher_output_{timestamp}\n\n{content}"
    with open(filename, "a") as f:
        f.write(formatted_text)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    # Split content into lines and add to PDF
    lines = content.split('\n')
    for line in lines:
        pdf.multi_cell(0, 10, line)
    
    pdf.output(filename)
    return f" Data saved to {filename}"

save_pdf_tool = Tool(
    name="Save to PDF",
    func=save_to_pdf,
    description="Saves the research output to a PDF file."
)