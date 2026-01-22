/**
 * Icon Generator Script
 * Generates app icons for Windows, macOS, and Linux from PNG source
 *
 * Usage: node scripts/generate-icons.js
 *
 * Requirements:
 * - npm install sharp png-to-ico icns-lib
 *
 * Note: Uses icon.png (1024x1024) as the source for best quality
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

const SOURCE_PNG = path.join(__dirname, '../public/icon.png');
const OUTPUT_DIR = path.join(__dirname, '../public');

// Icon sizes needed
const SIZES = {
  png: [16, 32, 48, 64, 128, 256, 512, 1024],
  ico: [16, 24, 32, 48, 64, 128, 256],
};

async function generateIcons() {
  console.log('Generating app icons...\n');

  // Check if source PNG exists
  if (!fs.existsSync(SOURCE_PNG)) {
    console.error(`Error: Source icon not found at ${SOURCE_PNG}`);
    console.error('Please ensure icon.png (1024x1024 recommended) exists in public/');
    process.exit(1);
  }

  // Read source PNG
  const sourceBuffer = fs.readFileSync(SOURCE_PNG);
  const metadata = await sharp(sourceBuffer).metadata();
  console.log(`Source: icon.png (${metadata.width}x${metadata.height})\n`);

  // Generate PNG files at various sizes (skip sizes larger than source)
  console.log('Generating PNG icons...');
  for (const size of SIZES.png) {
    if (size > metadata.width) {
      console.log(`  Skipped: icon-${size}.png (larger than source)`);
      continue;
    }
    const outputPath = path.join(OUTPUT_DIR, `icon-${size}.png`);
    await sharp(sourceBuffer)
      .resize(size, size, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .png()
      .toFile(outputPath);
    console.log(`  Created: icon-${size}.png`);
  }

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
