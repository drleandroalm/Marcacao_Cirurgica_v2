# Product Overview

SwiftTranscriptionSampleApp is an AI-powered medical form automation system that transforms speech into structured surgical scheduling forms. Originally based on Apple's WWDC25 SpeechAnalyzer sample, it has evolved into a production-ready medical solution for Brazilian healthcare.

## Core Purpose
- Convert continuous Portuguese speech into structured surgical request forms
- Support both field-by-field and continuous dictation modes
- Achieve 99.9% accuracy for known medical entities through AI-powered extraction

## Key Features
- **Dual Recording Modes**: Sequential field input or continuous one-take recording
- **AI Entity Extraction**: Uses iOS 26 Foundation Models for intelligent text processing
- **Medical Validation**: Whitelist system for surgeons and procedures with fuzzy matching
- **Portuguese Optimization**: Specialized for Brazilian medical terminology and expressions
- **Smart Formatting**: Automatic military time conversion, phone number formatting, date parsing

## Target Users
Brazilian healthcare professionals who need to quickly create surgical scheduling forms through voice dictation.

## Medical Context
The app generates "SOLICITAÇÃO DE AGENDAMENTO CIRÚRGICO" (Surgical Scheduling Request) forms with patient data, surgeon information, procedure details, and post-transcription decisions (CTI needs, patient precautions, OPME requirements).