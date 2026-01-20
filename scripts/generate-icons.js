/**
 * Icon Generator Script
 * Generates app icons for Windows, macOS, and Linux from SVG source
 *
 * Usage: node scripts/generate-icons.js
 *
 * Requirements:
 * - npm install sharp png-to-ico icns-lib
 */

const fs = require('fs');
const path = require('path');

// Check if sharp is installed
let sharp;
try {
  sharp = require('sharp');
} catch (e) {
  console.log('Installing required dependencies...');
  require('child_process').execSync('npm install sharp --save-dev', { stdio: 'inherit' });
  sharp = require('sharp');
}

const SVG_PATH = path.join(__dirname, '../public/icon.svg');
const OUTPUT_DIR = path.join(__dirname, '../public');

// Icon sizes needed
const SIZES = {
  png: [16, 32, 48, 64, 128, 256, 512, 1024],
  ico: [16, 24, 32, 48, 64, 128, 256],
};

async function generateIcons() {
  console.log('Generating app icons...\n');

  // Read SVG
  const svgBuffer = fs.readFileSync(SVG_PATH);

  // Generate PNG files at various sizes
  console.log('Generating PNG icons...');
  for (const size of SIZES.png) {
    const outputPath = path.join(OUTPUT_DIR, `icon-${size}.png`);
    await sharp(svgBuffer)
      .resize(size, size)
      .png()
      .toFile(outputPath);
    console.log(`  Created: icon-${size}.png`);
  }

  // Generate main icon.png (512x512 for Linux)
  const mainPngPath = path.join(OUTPUT_DIR, 'icon.png');
  await sharp(svgBuffer)
    .resize(512, 512)
    .png()
    .toFile(mainPngPath);
  console.log('  Created: icon.png (512x512)');

  // Generate Windows ICO
  console.log('\nGenerating Windows ICO...');
  try {
    // png-to-ico uses ES modules, so we need dynamic import
    const pngToIcoModule = await import('png-to-ico');
    const pngToIco = pngToIcoModule.default;

    // png-to-ico expects file paths, so use the generated PNG files
    const icoSizes = [16, 32, 48, 64, 128, 256];
    const pngPaths = icoSizes.map(size => path.join(OUTPUT_DIR, `icon-${size}.png`));

    const icoBuffer = await pngToIco(pngPaths);
    fs.writeFileSync(path.join(OUTPUT_DIR, 'icon.ico'), icoBuffer);
    console.log('  Created: icon.ico');
  } catch (e) {
    console.log('  Error generating ICO:', e.message);
    console.log('  If png-to-ico is not installed, run: npm install png-to-ico --save-dev');
  }

  // Generate macOS ICNS
  console.log('\nGenerating macOS ICNS...');
  try {
    const icnsLib = require('icns-lib');

    // ICNS icon types and their corresponding sizes
    // ic07 = 128x128, ic08 = 256x256, ic09 = 512x512, ic10 = 1024x1024
    // ic11 = 32x32 (16@2x), ic12 = 64x64 (32@2x), ic13 = 256x256 (128@2x), ic14 = 512x512 (256@2x)
    const iconTypes = {
      'ic07': 128,   // 128x128
      'ic08': 256,   // 256x256
      'ic09': 512,   // 512x512
      'ic10': 1024,  // 1024x1024 (512@2x)
      'ic11': 32,    // 32x32 (16@2x)
      'ic12': 64,    // 64x64 (32@2x)
      'ic13': 256,   // 256x256 (128@2x)
      'ic14': 512,   // 512x512 (256@2x)
    };

    const icons = {};
    for (const [type, size] of Object.entries(iconTypes)) {
      const pngPath = path.join(OUTPUT_DIR, `icon-${size}.png`);
      icons[type] = fs.readFileSync(pngPath);
    }

    // Create ICNS buffer
    const icnsBuffer = icnsLib.format(icons);
    fs.writeFileSync(path.join(OUTPUT_DIR, 'icon.icns'), icnsBuffer);
    console.log('  Created: icon.icns');
  } catch (e) {
    console.log('  Error generating ICNS:', e.message);
    console.log('  If icns-lib is not installed, run: npm install icns-lib --save-dev');
    console.log('\n  Alternative: Use online converter (cloudconvert.com, iconverticons.com)');
  }

  console.log('\nIcon generation complete!');
}

generateIcons().catch(console.error);
