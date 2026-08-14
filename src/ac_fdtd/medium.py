"""Properties of the propagation medium.

Only the two numbers the lossless scheme needs live here — sound speed and density. The
frequency-dependent part of air (the ISO 9613-1 absorption, which depends on humidity as much
as on temperature) is a separate concern and gets its own module; this one stays small so that
the scheme can be run in an arbitrary fluid without dragging in an air model.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Medium", "AIR"]

#: Specific gas constant of dry air, J/(kg·K).
_R_DRY_AIR = 287.058
_ZERO_CELSIUS_IN_KELVIN = 273.15


@dataclass(frozen=True)
class Medium:
    """A homogeneous, non-dispersive fluid.

    Attributes:
        sound_speed: c, in m/s.
        density: rho, in kg/m^3.
    """

    sound_speed: float
    density: float

    @property
    def impedance(self) -> float:
        """Characteristic specific acoustic impedance rho*c, in Pa·s/m.

        This is the reference against which wall admittances are expressed, so it is worth
        having as a named quantity rather than an inline product.
        """
        return self.density * self.sound_speed

    @classmethod
    def air(
        cls,
        temperature: float = 20.0,
        static_pressure: float = 101325.0,
    ) -> Medium:
        """Air at a given temperature (°C) and static pressure (Pa).

        Humidity is deliberately ignored here: it shifts the sound speed by well under 0.1 %
        at ordinary conditions, which is far below the discretisation error, while its real
        effect — absorption — is several orders of magnitude larger and belongs to the air
        absorption model rather than to c.
        """
        kelvin = temperature + _ZERO_CELSIUS_IN_KELVIN
        sound_speed = 331.3 * (kelvin / _ZERO_CELSIUS_IN_KELVIN) ** 0.5
        density = static_pressure / (_R_DRY_AIR * kelvin)
        return cls(sound_speed=sound_speed, density=density)


#: Air at 20 °C and one standard atmosphere: c = 343.2 m/s, rho = 1.204 kg/m^3.
AIR = Medium.air()
