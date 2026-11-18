from ukrainian_word_stress import Stressifier, StressSymbol

stressifier = Stressifier(
    stress_symbol=StressSymbol.CombiningAcuteAccent
)

text = "Мій перший тестовий приклад"

result = stressifier(text)

print(result)