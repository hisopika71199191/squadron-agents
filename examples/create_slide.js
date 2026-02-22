const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.title = 'Quick Test Presentation';

let slide = pres.addSlide();
slide.background = { color: "F1F1F1" };

// Add title
slide.addText("Quick Test Slide", {
  x: 0.5,
  y: 0.5,
  w: 9,
  h: 1,
  fontSize: 44,
  fontFace: "Arial",
  bold: true,
  color: "1E2761",
  align: "center"
});

// Add content
slide.addText([
  { text: "This is a test slide", options: { breakLine: true, fontSize: 24, color: "363636" } },
  { text: "Created with pptxgenjs", options: { breakLine: true, fontSize: 24, color: "363636" } },
  { text: "Saved as quick_test.pptx", options: { fontSize: 24, color: "363636" } }
], {
  x: 1,
  y: 2,
  w: 8,
  h: 3,
  align: "center"
});

pres.writeFile({ fileName: "quick_test.pptx" })
  .then(fileName => console.log("Presentation saved:", fileName))
  .catch(err => console.error("Error:", err));