/**
 * Syncs the version in package.json from the git tag.
 * Works both locally (git describe) and in GitHub Actions (GITHUB_REF_NAME).
 */
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const packageJsonPath = path.join(__dirname, '..', 'package.json');

function getVersionFromTag() {
  // In GitHub Actions, use GITHUB_REF_NAME when triggered by a tag
  // GITHUB_REF_NAME will be "v1.2.1" for tag triggers
  const githubRef = process.env.GITHUB_REF_NAME;
  if (githubRef && githubRef.match(/^v?\d+\.\d+\.\d+/)) {
    console.log(`Using GitHub ref: ${githubRef}`);
    return githubRef;
  }

  // Fallback: try git describe for local builds
  try {
    const tag = execSync('git describe --tags --abbrev=0', { encoding: 'utf8' }).trim();
    console.log(`Using git tag: ${tag}`);
    return tag;
  } catch {
    return null;
  }
}

try {
  const tag = getVersionFromTag();

  if (!tag) {
    console.log('No git tag found, keeping existing version');
    process.exit(0);
  }

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
  console.error('Failed to sync version:', error.message);
  process.exit(1);
}
