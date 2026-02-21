# Windows Installer UI Mapping (Figma -> NSIS)

This document captures how the Windows installer maps the Figma `Custom Installer Animation.make` style into a practical NSIS implementation.

## Visual Language from Figma

- Background: deep navy / near-black (`#060912` family)
- Accent: electric blue glow (`#00AAFF`, `#00CCFF`)
- Typography intent:
  - display: Orbitron-like futuristic headings
  - body: Inter-like readable text
  - technical labels: JetBrains Mono style
- Interaction pattern:
  - step-based flow (welcome -> options -> install -> done)
  - explicit verification feedback

## Practical NSIS Mapping

NSIS does not support full custom React-like animation scenes out of the box.
The shipped implementation prioritizes reliability while preserving visual intent:

- branded title and subtitle on custom options page
- generated brand bitmaps from `packaging/windows/design-tokens.json`
- high-contrast deterministic copy
- optional PATH + optional strict doctor verification
- install log for auditability

## Why This Tradeoff

- Figma Make animation is ideal as a brand surface, but not an installer runtime.
- NSIS is battle-tested for executable install/uninstall and environment setup.
- We keep installation logic in NSIS (stable) and reserve animation-rich surfaces for launcher/onboarding UI in v2.

## V2 UI Upgrade Path

- export Figma assets (header/welcome bitmaps, icons) into `packaging/windows/assets/`
- keep color/text source of truth in `packaging/windows/design-tokens.json`
- wire branded assets into NSIS `MUI_HEADERIMAGE` / welcome page visuals
- optional native custom installer shell (WinUI/WPF) that delegates install operations to the same verified core commands
