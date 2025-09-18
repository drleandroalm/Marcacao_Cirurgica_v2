import Foundation

// Test transcript in Portuguese with all 8 fields
let testTranscript = """
Paciente João Silva Santos, 45 anos, telefone 11987654321.
Cirurgia marcada para amanhã às 14 horas.
Procedimento será uma apendicectomia laparoscópica.
Doutor Pedro Almeida será o cirurgião responsável.
Tempo estimado do procedimento é de 2 horas.
"""

print("🧪 TEST: Entity Extraction from Sample Transcript")
print("=" * 50)
print("📝 Test Transcript:")
print(testTranscript)
print("=" * 50)

// Expected extractions:
// 1. patientName: "João Silva Santos"
// 2. patientAge: "45" 
// 3. patientPhone: "11987654321"
// 4. surgeryDate: [tomorrow's date]
// 5. surgeryTime: "14:00"
// 6. procedureName: "apendicectomia laparoscópica"
// 7. surgeonName: "Pedro Almeida"
// 8. procedureDuration: "2 horas"

print("\n🎯 Expected Entities:")
print("1. Patient Name: João Silva Santos")
print("2. Patient Age: 45")
print("3. Patient Phone: 11987654321")
print("4. Surgery Date: [tomorrow]")
print("5. Surgery Time: 14:00")
print("6. Procedure: apendicectomia laparoscópica")
print("7. Surgeon: Pedro Almeida")
print("8. Duration: 2 horas")
print("\n📊 Total Expected: 8 entities")