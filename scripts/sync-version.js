/**
 * Syncs the version in package.json from the latest git tag.
 * Run this before building to ensure the app version matches the release tag.
 */
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const packageJsonPath = path.join(__dirname, '..', 'package.json');

try {
  // Get the latest git tag (e.g., "v1.2.1")
  const tag = execSync('git describe --tags --abbrev=0', { encoding: 'utf8' }).trim();

  // Remove the 'v' prefix if present (v1.2.1 -> 1.2.1)
  const version = tag.startsWith('v') ? tag.slice(1) : tag;

  // Read and update package.json
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  const oldVersion = packageJson.version;

  if (oldVersion === version) {
    console.log(`Version already at ${version}`);
  } else {
    packageJson.version = version;
    fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2) + '\n');
    console.log(`Updated version: ${oldVersion} -> ${version}`);
  }
} catch (error) {
  console.error('Failed to sync version from git tag:', error.message);
  console.log('Keeping existing version in package.json');
}
