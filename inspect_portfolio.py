
from bs4 import BeautifulSoup

# Read downloaded HTML
with open("data/portfolio.html", "r", encoding="utf-8") as file:
    html = file.read()

soup = BeautifulSoup(html, "html.parser")

# Remove unnecessary elements
for element in soup(["script", "style", "noscript", "svg"]):
    element.decompose()

# Extract page title
print("\n========== PAGE TITLE ==========")
print(soup.title.get_text(strip=True) if soup.title else "No title")

# Extract headings
print("\n========== HEADINGS ==========")
for heading in soup.find_all(["h1", "h2", "h3"]):
    text = heading.get_text(" ", strip=True)
    if text:
        print(f"[{heading.name}] {text}")

# Extract readable body text
print("\n========== PORTFOLIO CONTENT ==========")
text = soup.body.get_text(separator="\n", strip=True) if soup.body else soup.get_text("\n", strip=True)

# Remove empty lines
lines = [line.strip() for line in text.splitlines() if line.strip()]
clean_text = "\n".join(lines)

# Save extracted content
with open("data/portfolio_text.txt", "w", encoding="utf-8") as file:
    file.write(clean_text)

print(clean_text)

print("\n========== EXTRACTION SUMMARY ==========")
print("Total characters:", len(clean_text))
print("Total lines:", len(lines))
print("Saved to: data/portfolio_text.txt")
