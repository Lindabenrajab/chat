import re
import json
from urllib.parse import urlparse, urljoin
import streamlit as st
from langchain_core.documents import Document

# --- Internal link extraction ---
def extract_internal_links_bs(html, base_url):
    from bs4 import BeautifulSoup
    base_domain = urlparse(base_url).netloc
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        full_url = urljoin(base_url, href)
        parsed_url = urlparse(full_url)
        # Add all internal links from the same domain, except root and empty
        if parsed_url.netloc == base_domain and parsed_url.path not in ['/', '']:
            links.add(full_url)
    return list(links)

# --- Document processing ---
def process_crawled_content(extracted_content, url):
    documents = []
    try:
        if isinstance(extracted_content, str):
            content_data = json.loads(extracted_content)
        else:
            content_data = extracted_content
        if "sections" in content_data:
            for section in content_data["sections"]:
                if section.get("relevance") in ["high", "medium"] and section.get("content"):
                    doc = Document(
                        page_content=section["content"],
                        metadata={
                            "question": section.get("title", "Web Content"),
                            "source": "Web",
                            "type": "dynamic",
                            "url": url,
                            "relevance": section.get("relevance", "medium")
                        }
                    )
                    documents.append(doc)
    except json.JSONDecodeError as e:
        st.warning(f"[Diagnostics] JSON decode error processing content from {url}: {str(e)}")
    except Exception as e:
        st.warning(f"[Diagnostics] Error processing content from {url}: {str(e)}")
    return documents

# --- robots.txt check ---
def is_allowed_by_robots(target_url):
    import urllib.robotparser
    parsed = urlparse(target_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch('*', target_url)
        if not allowed:
            st.warning(f"[Diagnostics] robots.txt disallows crawling: {target_url}")
        return allowed
    except Exception as e:
        st.info(f"Could not fetch robots.txt for {target_url}: {e}")
        return True
