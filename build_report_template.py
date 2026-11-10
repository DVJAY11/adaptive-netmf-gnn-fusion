from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = Path("artifacts/final_project_report_template.docx")
OUT.parent.mkdir(parents=True, exist_ok=True)
doc = Document()
sec = doc.sections[0]
sec.top_margin = sec.bottom_margin = Inches(0.75)
sec.left_margin = sec.right_margin = Inches(0.85)
styles = doc.styles
styles["Normal"].font.name = "Aptos"
styles["Normal"].font.size = Pt(11)
for name, size in (("Title", 22), ("Heading 1", 16), ("Heading 2", 13)):
    styles[name].font.name = "Aptos Display"
    styles[name].font.size = Pt(size)
    styles[name].font.color.rgb = RGBColor(0, 0, 0)

title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.add_run("Adaptive Fusion of NetMF and Graph Neural Networks")
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("Final Project Report Template\n").bold = True
p.add_run("CSC 591 791 Machine Learning with Graphs\nDigvijay and Team")

doc.add_heading("Abstract", 1)
doc.add_paragraph(
    "We study whether combining NetMF with message passing graph neural networks improves node classification under limited labels and missing edges. We compare individual models, a GCN that receives NetMF embeddings as extra features, simple prediction averaging, and a learned router. [INSERT FINAL RESULT AND CONCLUSION AFTER FIVE SEED RUNS.]"
)

sections = {
"Problem and Motivation": "Cora and CiteSeer represent papers as nodes and citations as edges. The task is to predict each paper's topic. NetMF summarizes multi-hop structure, while GCN and GraphSAGE combine node features with local neighborhoods. The project tests whether their different predictions provide useful complementary information.",
"Research Question": "Does adaptive prediction fusion improve node classification compared with a standard GCN and the simpler approach of concatenating NetMF embeddings with GCN input features?",
"Datasets": "Cora contains 2,708 papers, 1,433 word features, and seven classes. CiteSeer provides a second small citation network when the optional run is completed. We use transductive node classification and keep the test labels untouched until evaluation.",
"Experimental Protocol": "For each seed, we sample 5, 10, or 20 training nodes per class. We divide the official validation nodes by class: 50 percent stop the base models, 30 percent train the router, and 20 percent stop the router. Final results use seeds 42 through 46. The robustness experiment removes 0, 20, or 40 percent of edges and recomputes NetMF on the same damaged graph.",
"Methods": "The baselines are MLP, NetMF with logistic regression, GCN, and GraphSAGE. NetMF plus GCN appends the NetMF embedding to each node's original features. Equal averaging combines the three expert probabilities. A fixed router uses manual degree weights. The learned router predicts weights from degree, confidence, entropy, and model disagreement. Degree-only and confidence-only variants serve as ablations.",
"Oracle Analysis": "The oracle counts a prediction as correct when any expert predicts the true class. It measures possible routing headroom but cannot be deployed because it uses the true label to select an expert.",
}
for heading, text in sections.items():
    doc.add_heading(heading, 1)
    doc.add_paragraph(text)

doc.add_heading("Final Results", 1)
doc.add_paragraph("Replace bracketed cells only with results from the completed five-seed experiment.")
table = doc.add_table(rows=1, cols=6)
table.style = "Table Grid"
headers = ["Method", "5 labels", "10 labels", "20 labels", "Runtime", "GPU memory"]
for cell, text in zip(table.rows[0].cells, headers):
    cell.text = text
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    shading = OxmlElement("w:shd"); shading.set(qn("w:fill"), "1F4E78"); cell._tc.get_or_add_tcPr().append(shading)
    for run in cell.paragraphs[0].runs:
        run.font.color.rgb = RGBColor(255, 255, 255); run.bold = True
for method in ["GCN", "NetMF plus GCN", "Equal average", "Learned router"]:
    cells = table.add_row().cells
    cells[0].text = method
    for cell in cells[1:]: cell.text = "[INSERT]"

for heading, text in [
    ("Degree Group Analysis", "[INSERT LOW, MEDIUM, AND HIGH DEGREE RESULTS.] Explain whether the original degree hypothesis is supported rather than assuming that NetMF helps low-degree nodes."),
    ("Missing Edge Results", "[INSERT 0, 20, AND 40 PERCENT EDGE REMOVAL RESULTS.] Compare every method on the same damaged graph."),
    ("Discussion", "Compare the learned router first against GCN and NetMF plus GCN. Prefer the simpler model when accuracy is similar. Use the oracle gap to explain whether additional router research has room to help."),
    ("Limitations", "The study uses small citation networks in a transductive setting. NetMF also depends on connectivity, so low degree does not imply that it will perform well. Five seeds measure variation but do not establish broad statistical significance. Explicit NetMF does not scale to very large graphs."),
    ("Conclusion", "[INSERT THE FINAL EVIDENCE-BASED CONCLUSION.] State clearly whether adaptive fusion justified its additional computation."),
]:
    doc.add_heading(heading, 1); doc.add_paragraph(text)

doc.save(OUT)
print(OUT.resolve())
