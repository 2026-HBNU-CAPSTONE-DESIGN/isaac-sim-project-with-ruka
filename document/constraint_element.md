# 기존에서 변경된 제한 및 확인 필요 사항

## URDF 파일 변경

- Palm 질량은 임시값 (0.05 kg, default inertial). MJCF에 Palm의 inertial이 없어서 작은 기본값을 넣었습니다. Isaac에서 동작 확인 후 실측값이 필요하면 따로 측정하거나 USD 단계에서 조정.

- 관절 effort/velocity 한계는 임시값 (10/10). RUKA가 텐던 구동이라 정확한 모터 토크는 다르지만, Isaac Lab의 ArticulationCfg에서 어차피 actuator를 따로 정의하므로 URDF 값은 학습 코드에서 덮어쓰기됩니다.

- Collision은 box primitive로 변환 (mesh 아님). MJCF가 그렇게 정의했고, 학습 안정성에 좋습니다.

- TIP 마커는 작은 sphere visual로 유지. fingertip 위치 추적이나 관측에 쓸 수 있어요.
