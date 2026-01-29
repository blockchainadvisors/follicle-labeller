# Chapter 1: Project Overview

## 1.1 Introduction

The Follicle Labeller is a desktop application designed for medical image annotation, specifically targeting the labeling and detection of hair follicles in dermatological images. This project was developed during a master's placement at Blockchain Advisors, combining computer vision, machine learning, and modern desktop application development.

## 1.2 Problem Statement

Medical image annotation is a time-consuming and labor-intensive process that requires precise marking of features within images. In dermatology, particularly for hair follicle analysis, researchers and clinicians need to:

1. **Manually annotate** thousands of follicles across multiple images
2. **Maintain consistency** in annotation standards
3. **Export data** in formats suitable for machine learning training
4. **Collaborate** across different platforms (Windows, macOS, Linux)

The lack of specialized tools for follicle annotation leads to:
- Inconsistent annotation quality
- Slow annotation throughput
- Difficulty in training machine learning models
- Limited automation options

## 1.3 Project Goals

### Primary Objectives

1. **Create a cross-platform desktop application** for hair follicle annotation
2. **Implement multiple annotation shapes** (circles, rectangles, oriented boxes)
3. **Add automated detection capabilities** using computer vision and ML
4. **Enable ML model training** directly from the application
5. **Support multi-image projects** with proper state management

### Secondary Objectives

1. Implement GPU acceleration for detection algorithms
2. Create a comprehensive design system for UI consistency
3. Support automatic updates across all platforms
4. Enable annotation import/export in multiple formats

## 1.4 Scope

### In Scope

| Feature | Description |
|---------|-------------|
| Annotation Tools | Circle, Rectangle, Linear (Oriented Box) shapes |
| Detection Methods | BLOB detection (OpenCV), YOLO11 detection, YOLO keypoint |
| File Formats | `.fol` project format, JSON, CSV, YOLO dataset export |
| Platforms | Windows, macOS (Intel + Apple Silicon), Linux |
| GPU Support | NVIDIA CUDA, Apple Metal/MPS |
| ML Training | YOLO11 model training with resume capability |

### Out of Scope

- Web-based version of the application
- Real-time collaboration features
- Cloud storage integration
- Mobile application support

## 1.5 Deliverables

### Software Deliverables

1. **Follicle Labeller Application** (v2.0.2)
   - Windows installer (NSIS)
   - macOS DMG (Universal Binary)
   - Linux AppImage

2. **Python Backend Services**
   - BLOB detection server
   - YOLO keypoint training service
   - YOLO detection training service

3. **Documentation**
   - User README with installation instructions
   - FOL file format specification
   - UI/UX design guidelines

### Technical Deliverables

1. Complete source code with TypeScript/React frontend
2. Python services with FastAPI backend
3. Automated CI/CD pipeline for releases
4. Comprehensive test suite

## 1.6 Project Timeline Summary

```
Week 1 (Jan 12-14): Foundation & Core Features
├── Initial Electron app setup
├── Basic annotation tools (circle, rectangle)
├── File save/load functionality
└── Auto-update system

Week 2 (Jan 20-21): Feature Expansion
├── Multi-selection feature
├── Image extraction and download
├── BLOB detection integration
├── Theme customization

Week 3 (Jan 22-26): Platform & Detection
├── macOS code signing and notarization
├── GPU acceleration (CUDA, Metal)
├── Advanced BLOB detection parameters
├── FastAPI migration

Week 4 (Jan 27-29): ML & Polish
├── YOLO11 keypoint training
├── YOLO11 detection training
├── Annotation import/export
├── UI/UX consistency
└── Final bug fixes
```

## 1.7 Key Stakeholders

| Role | Responsibility |
|------|----------------|
| Development Lead | Overall project direction, code review |
| Master's Student (Laurentiu Nae) | Primary development, implementation |
| Team Members (Ali Cagatay, kechyy) | Feature contributions, bug fixes |
| End Users | Medical researchers, dermatologists |

## 1.8 Success Criteria

1. **Functional**: All core annotation and detection features working
2. **Performance**: Detection completes in reasonable time on large images
3. **Reliability**: No data loss, proper error handling
4. **Usability**: Intuitive UI with keyboard shortcuts
5. **Portability**: Works on all three major desktop platforms
