const pptxgen = require("pptxgenjs");

async function createPresentation() {
  let pres = new pptxgen();
  pres.layout = 'LAYOUT_16x9';
  pres.author = 'Game Recommendations Agent';
  pres.title = 'Top 3 Online Games 2026';

  // Color palette: Ocean Gradient theme
  const colors = {
    primary: '065A82',
    secondary: '1C7293',
    accent: '21295C',
    light: 'E8F4F8',
    white: 'FFFFFF'
  };

  // Slide 1: Title Slide
  let titleSlide = pres.addSlide();
  titleSlide.background = { color: colors.primary };
  titleSlide.addText("Top 3 Online Games", {
    x: 0.5, y: 1.5, w: 9, h: 1,
    fontSize: 44, fontFace: "Arial Black",
    color: colors.white, bold: true, align: "center"
  });
  titleSlide.addText("Recommendations for 2026", {
    x: 0.5, y: 2.5, w: 9, h: 0.8,
    fontSize: 24, fontFace: "Arial",
    color: colors.light, align: "center"
  });
  titleSlide.addShape(pres.shapes.OVAL, {
    x: 4.5, y: 3.5, w: 1, h: 1,
    fill: { color: colors.secondary }
  });

  // Slide 2: Games 1 & 2
  let gamesSlide1 = pres.addSlide();
  gamesSlide1.background = { color: colors.light };
  gamesSlide1.addText("Featured Games #1-2", {
    x: 0.5, y: 0.3, w: 9, h: 0.6,
    fontSize: 36, fontFace: "Arial Black",
    color: colors.accent, bold: true
  });
  
  // Game 1 card
  gamesSlide1.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 1.2, w: 4.25, h: 3.5,
    fill: { color: colors.white },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.15 }
  });
  gamesSlide1.addText("1. Starfield: Shattered Space", {
    x: 0.7, y: 1.4, w: 3.8, h: 0.4,
    fontSize: 18, fontFace: "Arial",
    color: colors.primary, bold: true
  });
  gamesSlide1.addText(["Genre: Space Exploration MMORPG"], {
    x: 0.7, y: 1.9, w: 3.8, h: 0.3,
    fontSize: 14, fontFace: "Arial", color: "444444"
  });
  gamesSlide1.addText(["Why: Bethesda's expansion into persistent online universe."], {
    x: 0.7, y: 2.3, w: 3.8, h: 0.5,
    fontSize: 12, fontFace: "Arial", color: "555555"
  });
  gamesSlide1.addText(["Key: Seamless planetary exploration with thousands of concurrent players."], {
    x: 0.7, y: 2.9, w: 3.8, h: 0.5,
    fontSize: 12, fontFace: "Arial", color: "555555"
  });

  // Game 2 card
  gamesSlide1.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 5.25, y: 1.2, w: 4.25, h: 3.5,
    fill: { color: colors.white },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.15 }
  });
  gamesSlide1.addText("2. The Elder Scrolls VI: Online", {
    x: 5.45, y: 1.4, w: 3.8, h: 0.4,
    fontSize: 18, fontFace: "Arial",
    color: colors.primary, bold: true
  });
  gamesSlide1.addText(["Genre: Fantasy MMORPG"], {
    x: 5.45, y: 1.9, w: 3.8, h: 0.3,
    fontSize: 14, fontFace: "Arial", color: "444444"
  });
  gamesSlide1.addText(["Why: Next evolution of Tamriel, built on new engine with cross-play."], {
    x: 5.45, y: 2.3, w: 3.8, h: 0.5,
    fontSize: 12, fontFace: "Arial", color: "555555"
  });
  gamesSlide1.addText(["Key: Living world with dynamic events and player-driven economy."], {
    x: 5.45, y: 2.9, w: 3.8, h: 0.5,
    fontSize: 12, fontFace: "Arial", color: "555555"
  });

  // Slide 3: Game 3 + Conclusion
  let gamesSlide2 = pres.addSlide();
  gamesSlide2.background = { color: colors.light };
  gamesSlide2.addText("Featured Game #3", {
    x: 0.5, y: 0.3, w: 9, h: 0.6,
    fontSize: 36, fontFace: "Arial Black",
    color: colors.accent, bold: true
  });
  
  // Game 3 card
  gamesSlide2.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 1.2, w: 9, h: 2.5,
    fill: { color: colors.white },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.15 }
  });
  gamesSlide2.addText("3. Valorant: Global Tactics", {
    x: 0.7, y: 1.4, w: 8.6, h: 0.4,
    fontSize: 20, fontFace: "Arial",
    color: colors.primary, bold: true
  });
  gamesSlide2.addText(["Genre: Tactical Shooter Battle Royale"], {
    x: 0.7, y: 1.95, w: 8.6, h: 0.3,
    fontSize: 14, fontFace: "Arial", color: "444444"
  });
  gamesSlide2.addText(["Why: Riot's evolution combining tactical FPS with large-scale BR mechanics."], {
    x: 0.7, y: 2.4, w: 8.6, h: 0.4,
    fontSize: 12, fontFace: "Arial", color: "555555"
  });
  gamesSlide2.addText(["Key: 60-player matches with agent abilities and strategic zone control."], {
    x: 0.7, y: 2.9, w: 8.6, h: 0.4,
    fontSize: 12, fontFace: "Arial", color: "555555"
  });

  // Footer
  gamesSlide2.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.2, w: 10, h: 0.425,
    fill: { color: colors.accent }
  });
  gamesSlide2.addText("Generated for 2026 Gaming Trends | Top Games Recommendation", {
    x: 0.5, y: 5.3, w: 9, h: 0.3,
    fontSize: 10, fontFace: "Arial",
    color: colors.light, align: "center"
  });

  await pres.writeFile({ fileName: "top_games_2026.pptx" });
  console.log("Presentation created successfully!");
}

createPresentation().catch(console.error);
