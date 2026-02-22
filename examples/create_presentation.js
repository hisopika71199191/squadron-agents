const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.title = 'Top 3 Online Games 2026';
pres.author = 'AI Assistant';

// Define a modern color palette: Ocean Gradient theme
const colors = {
  primary: "065A82",   // Deep Blue
  secondary: "1C7293", // Teal
  accent: "FFFFFF",    // White
  textDark: "21295C",  // Midnight
  textLight: "F0F4F8"  // Light Blue-Gray
};

// Slide 1: Title Slide
let slide1 = pres.addSlide();
slide1.background = { color: colors.primary };
slide1.addText("Top 3 Online Games\nof 2026", {
  x: 1, y: 1.5, w: 8, h: 2,
  fontSize: 44, fontFace: "Arial Black",
  color: colors.accent, bold: true, align: "center"
});
slide1.addText("Future-Proof Gaming Recommendations", {
  x: 1, y: 3.5, w: 8, h: 0.5,
  fontSize: 18, fontFace: "Arial",
  color: colors.secondary, align: "center", italic: true
});
slide1.addShape(pres.shapes.LINE, {
  x: 3, y: 4.2, w: 4, h: 0,
  line: { color: colors.secondary, width: 3 }
});

// Slide 2: Top 2 Games (Grid Layout)
let slide2 = pres.addSlide();
slide2.background = { color: "F0F4F8" };
slide2.addText("#1 & #2 Recommendations", {
  x: 0.5, y: 0.3, w: 9, h: 0.5,
  fontSize: 32, fontFace: "Arial Black",
  color: colors.textDark, bold: true
});

// Game 1 Card
slide2.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.5, y: 1.0, w: 4.25, h: 3.5,
  fill: { color: "FFFFFF" },
  rectRadius: 0.1,
  shadow: { type: "outer", color: "000000", blur: 10, offset: 4, angle: 135, opacity: 0.1 }
});
slide2.addText("1. Grand Theft Auto VI Online", {
  x: 0.7, y: 1.2, w: 3.8, h: 0.4,
  fontSize: 20, fontFace: "Arial",
  color: colors.primary, bold: true
});
slide2.addText("• Immersive open-world multiplayer\n• Dynamic economy & heists\n• Cross-platform integration", {
  x: 0.7, y: 1.7, w: 3.8, h: 2.5,
  fontSize: 14, fontFace: "Calibri",
  color: "444444",
  bullet: true, breakLine: true
});

// Game 2 Card
slide2.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 5.25, y: 1.0, w: 4.25, h: 3.5,
  fill: { color: "FFFFFF" },
  rectRadius: 0.1,
  shadow: { type: "outer", color: "000000", blur: 10, offset: 4, angle: 135, opacity: 0.1 }
});
slide2.addText("2. The Elder Scrolls VI: Online", {
  x: 5.45, y: 1.2, w: 3.8, h: 0.4,
  fontSize: 20, fontFace: "Arial",
  color: colors.primary, bold: true
});
slide2.addText("• Massive fantasy MMORPG world\n• Guild wars & exploration\n• Next-gen graphics engine", {
  x: 5.45, y: 1.7, w: 3.8, h: 2.5,
  fontSize: 14, fontFace: "Calibri",
  color: "444444",
  bullet: true, breakLine: true
});

// Slide 3: #3 Game + Conclusion
let slide3 = pres.addSlide();
slide3.background = { color: "F0F4F8" };
slide3.addText("#3 Recommendation & Why These Games?", {
  x: 0.5, y: 0.3, w: 9, h: 0.5,
  fontSize: 28, fontFace: "Arial Black",
  color: colors.textDark, bold: true
});

// Game 3 Section
slide3.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.0, w: 9, h: 2.2,
  fill: { color: colors.secondary },
  rectRadius: 0
});
slide3.addText("3. Fortnite: Chapter 6 (Metaverse Hub)", {
  x: 0.7, y: 1.2, w: 8.6, h: 0.4,
  fontSize: 22, fontFace: "Arial",
  color: "FFFFFF", bold: true
});
slide3.addText("Evolved into a full social metaverse platform with concerts, creative modes, and competitive esports leagues dominating 2026.", {
  x: 0.7, y: 1.7, w: 8.6, h: 1.2,
  fontSize: 16, fontFace: "Calibri",
  color: "FFFFFF", align: "left"
});

// Key Trends Footer
slide3.addText("Key 2026 Trends:", {
  x: 0.5, y: 3.4, w: 2, h: 0.3,
  fontSize: 16, fontFace: "Arial",
  color: colors.textDark, bold: true
});
slide3.addText("Cross-Play • AI-Driven NPCs • Cloud Gaming • Social Integration", {
  x: 2.6, y: 3.4, w: 7, h: 0.3,
  fontSize: 14, fontFace: "Calibri",
  color: "555555", italic: true
});

pres.writeFile({ fileName: "Top_3_Online_Games_2026.pptx" })
  .then(fileName => console.log(`Presentation saved: ${fileName}`))
  .catch(err => console.error("Error:", err));
