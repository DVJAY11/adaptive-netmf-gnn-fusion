import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const outDir = "artifacts";
const previewDir = ".pptx-qa";
await fs.mkdir(outDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const slides = [
  ["Adaptive Fusion of NetMF and Graph Neural Networks", ["Final project presentation template", "CSC 591 791 Machine Learning with Graphs", "Digvijay and Team"], "Introduce the project and team. Final numerical claims will be added only after the five-seed runs."],
  ["Node classification problem", ["Each node is a research paper", "Each edge is a citation", "We predict the topic of every paper"], "Explain Cora in simple words. The model sees paper words and citation links, then predicts one of seven topics."],
  ["Research question", ["Can NetMF add information that GNNs miss?", "Does a learned router beat a standard GCN?", "Does routing justify the cost of running three experts?"], "The main comparison is not only router versus averaging. We must compare with the simpler NetMF plus GCN model."],
  ["Datasets and clean split", ["Cora: 2,708 nodes, 1,433 features, 7 classes", "Training budgets: 5, 10, and 20 nodes per class", "Validation: 50% base stopping, 30% router training, 20% router stopping", "Test labels remain untouched"], "Stress that the three validation subsets do not overlap. CiteSeer is the optional second dataset."],
  ["Methods", ["Baselines: MLP, NetMF, GCN, GraphSAGE", "Simple integration: NetMF embedding plus original features into GCN", "Prediction fusion: equal average and fixed degree router", "Adaptive method: learned router"], "Explain that NetMF plus GCN directly answers the professor's simpler alternative."],
  ["Learned router", ["Inputs: degree, confidence, entropy, and model disagreement", "A small MLP produces three positive weights", "Weighted probabilities produce the final class prediction", "Ablations test degree only, confidence only, and all inputs"], "The router learns which expert to trust for each node. It never receives test labels."],
  ["Oracle and ablations", ["Oracle: correct if any expert predicts correctly", "Oracle measures routing headroom only", "Ablations show which router inputs matter", "The fixed degree router remains a heuristic baseline"], "Do not present the oracle as a real method. It knows the true answer when choosing an expert."],
  ["Limited-label results", ["[INSERT FIVE-SEED ACCURACY PLOT]", "Compare GCN, NetMF plus GCN, and learned router", "Show mean and standard deviation"], "Use the final accuracy plot here. Explain whether the router beats both the strong baseline and the simpler integration."],
  ["Missing-edge results", ["[INSERT 0%, 20%, AND 40% EDGE-REMOVAL PLOT]", "Every method receives the same damaged graph", "NetMF is recomputed after edge removal"], "This is the robustness test. Avoid claiming that NetMF is naturally better on sparse graphs."],
  ["Degree-group results", ["[INSERT LOW, MEDIUM, AND HIGH DEGREE PLOT]", "Test the original degree hypothesis", "Report evidence instead of assuming NetMF helps low-degree nodes"], "This slide directly answers the professor's question about dependence on degree."],
  ["Accuracy and complexity", ["[INSERT FINAL RUNTIME AND GPU MEMORY TABLE]", "Prefer the simpler method when accuracy is similar", "The router runs NetMF, GCN, and GraphSAGE"], "Explain the practical tradeoff. A tiny router does not remove the cost of its three base experts."],
  ["Conclusion and limitations", ["[INSERT FINAL EVIDENCE-BASED CONCLUSION]", "Small transductive citation networks", "Five seeds measure variation but not broad significance", "Explicit NetMF does not scale to very large graphs"], "State the result honestly even if GCN remains best. A negative result can still answer the research question."],
];

function addText(slide, text, pos, size, color="#111827", bold=false) {
  const shape = slide.shapes.add({ geometry: "textbox", position: pos, fill: "none", line: { fill: "none", width: 0 } });
  shape.text = text;
  shape.text.style = { fontFamily: "Aptos", fontSize: size, color, bold };
  return shape;
}

for (let i = 0; i < slides.length; i++) {
  const [title, bullets, notes] = slides[i];
  const slide = deck.slides.add();
  slide.background.fill = i === 0 ? "#EFF6FF" : "#FFFFFF";
  addText(slide, title, { left: 72, top: i === 0 ? 150 : 58, width: 1136, height: i === 0 ? 150 : 64 }, i === 0 ? 48 : 34, "#123B5D", true);
  if (i === 0) {
    addText(slide, bullets.join("\n"), { left: 76, top: 340, width: 900, height: 170 }, 24, "#334155", false);
  } else {
    let top = 160;
    for (const bullet of bullets) {
      addText(slide, `•  ${bullet}`, { left: 100, top, width: 1080, height: 62 }, 24, bullet.startsWith("[") ? "#B45309" : "#1F2937", bullet.startsWith("["));
      top += 86;
    }
    addText(slide, `${i + 1}`, { left: 1160, top: 660, width: 50, height: 24 }, 12, "#64748B");
  }
  slide.speakerNotes.text = notes;
  const png = await deck.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(`${previewDir}/slide-${String(i + 1).padStart(2, "0")}.png`, new Uint8Array(await png.arrayBuffer()));
}

const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(`${outDir}/final_project_presentation_template.pptx`);
