import pyneb as pn

S2 = pn.Atom('S', 2)

print(f"The high density ratio is {S2.getHighDensRatio(wave1 = 6716, wave2 = 6731)}")
print(f"The low density ratio is {S2.getLowDensRatio(wave1 = 6716, wave2 = 6731)}")
print(f"{S2.getDensityRange(wave1 = 6716, wave2 = 6731)}")

print(f"Critical density for 6716 is {S2.getCritDensity(15000, 3)}")
print(f"Critical density for 6731 is {S2.getCritDensity(15000, 2)}")