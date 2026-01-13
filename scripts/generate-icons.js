/**
 * Icon Generator Script
 * Generates app icons for Windows, macOS, and Linux from SVG source
 *
 * Usage: node scripts/generate-icons.js
 *
 * Requirements:
 * - npm install sharp png-to-ico
 * - For macOS .icns: use online converter or macOS iconutil
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

  // Instructions for macOS ICNS
  console.log('\nFor macOS icon.icns:');
  console.log('  Option 1: Use online converter (cloudconvert.com, iconverticons.com)');
  console.log('  Option 2: On macOS, create iconset folder and use iconutil:');
  console.log('    mkdir icon.iconset');
  console.log('    cp icon-16.png icon.iconset/icon_16x16.png');
  console.log('    cp icon-32.png icon.iconset/icon_16x16@2x.png');
  console.log('    cp icon-32.png icon.iconset/icon_32x32.png');
  console.log('    cp icon-64.png icon.iconset/icon_32x32@2x.png');
  console.log('    cp icon-128.png icon.iconset/icon_128x128.png');
  console.log('    cp icon-256.png icon.iconset/icon_128x128@2x.png');
  console.log('    cp icon-256.png icon.iconset/icon_256x256.png');
  console.log('    cp icon-512.png icon.iconset/icon_256x256@2x.png');
  console.log('    cp icon-512.png icon.iconset/icon_512x512.png');
  console.log('    cp icon-1024.png icon.iconset/icon_512x512@2x.png');
  console.log('    iconutil -c icns icon.iconset -o icon.icns');

  console.log('\nIcon generation complete!');
}

generateIcons().catch(console.error);
