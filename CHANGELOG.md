# Changelog

## [0.2.14](https://github.com/BlueBrain/emodel-generalisation/compare/0.2.5..0.2.14)

> 20 October 2024

### New Features

- Improve readme (Alexis Arnaudon - [#55](https://github.com/BlueBrain/emodel-generalisation/pull/55))
- Can load .so mod (Alexis Arnaudon - [#54](https://github.com/BlueBrain/emodel-generalisation/pull/54))
- --no-reuse in cli (Alexis Arnaudon - [#52](https://github.com/BlueBrain/emodel-generalisation/pull/52))
- Migrate to simplified emodel recipe (Eleftherios Zisis - [#44](https://github.com/BlueBrain/emodel-generalisation/pull/44))
- Use BluePyParallel (Alexis Arnaudon - [#42](https://github.com/BlueBrain/emodel-generalisation/pull/42))

### Fixes

- Use Bluecellulab load mod wrapper (Alexis Arnaudon - [#60](https://github.com/BlueBrain/emodel-generalisation/pull/60))
- Do not load via bluecellulab (Alexis Arnaudon - [#58](https://github.com/BlueBrain/emodel-generalisation/pull/58))
- Fix no spike case when expected (Alexis Arnaudon - [#57](https://github.com/BlueBrain/emodel-generalisation/pull/57))
- Better handling of mechanism loading in cli (Alexis Arnaudon - [#50](https://github.com/BlueBrain/emodel-generalisation/pull/50))
- Use main evaluator in mcmc for consistency (Alexis Arnaudon - [#49](https://github.com/BlueBrain/emodel-generalisation/pull/49))
- Fix NO_PROGRESS (Alexis Arnaudon - [#47](https://github.com/BlueBrain/emodel-generalisation/pull/47))
- More fixing (Alexis Arnaudon - [#40](https://github.com/BlueBrain/emodel-generalisation/pull/40))
- One more params/parameters (Alexis Arnaudon - [#38](https://github.com/BlueBrain/emodel-generalisation/pull/38))
- Make sure param in final works (Alexis Arnaudon - [#36](https://github.com/BlueBrain/emodel-generalisation/pull/36))

### General Changes

- 0.2.13 (Alexis Arnaudon - [#59](https://github.com/BlueBrain/emodel-generalisation/pull/59))
- 0.2.12 (Alexis Arnaudon - [#56](https://github.com/BlueBrain/emodel-generalisation/pull/56))
- Minor updates (Alexis Arnaudon - [#51](https://github.com/BlueBrain/emodel-generalisation/pull/51))
- 0.2.11 (Alexis Arnaudon - [#53](https://github.com/BlueBrain/emodel-generalisation/pull/53))
- 0.2.10 (Alexis Arnaudon - [#48](https://github.com/BlueBrain/emodel-generalisation/pull/48))
- Fix dataframe dtype assignment warnings (Eleftherios Zisis - [#46](https://github.com/BlueBrain/emodel-generalisation/pull/46))
- 0.2.9 (Alexis Arnaudon - [#45](https://github.com/BlueBrain/emodel-generalisation/pull/45))
- 0.2.8 (Alexis Arnaudon - [#41](https://github.com/BlueBrain/emodel-generalisation/pull/41))
- 0.2.7 (Alexis Arnaudon - [#39](https://github.com/BlueBrain/emodel-generalisation/pull/39))
- Release 0.2.6 (Alexis Arnaudon - [#37](https://github.com/BlueBrain/emodel-generalisation/pull/37))

## [0.2.5](https://github.com/BlueBrain/emodel-generalisation/compare/0.2.3..0.2.5)

> 5 February 2024

### New Features

- Handle thalamus protocols (Alexis Arnaudon - [#22](https://github.com/BlueBrain/emodel-generalisation/pull/22))
- Add info theory module (Alexis Arnaudon - [#28](https://github.com/BlueBrain/emodel-generalisation/pull/28))
- Better config handling for combos (Alexis Arnaudon - [#30](https://github.com/BlueBrain/emodel-generalisation/pull/30))

### Fixes

- Local/nonlocal config folder for mapping (Alexis Arnaudon - [#32](https://github.com/BlueBrain/emodel-generalisation/pull/32))

### Chores And Housekeeping

- 0.2.4 (Alexis Arnaudon - [#31](https://github.com/BlueBrain/emodel-generalisation/pull/31))

### General Changes

- 0.2.5 (Alexis Arnaudon - [#34](https://github.com/BlueBrain/emodel-generalisation/pull/34))
- Release: 0.2.4 (Alexis Arnaudon - [#33](https://github.com/BlueBrain/emodel-generalisation/pull/33))

## [0.2.3](https://github.com/BlueBrain/emodel-generalisation/compare/0.2.2..0.2.3)

> 15 December 2023

### Fixes

- Better mech loading (Alexis Arnaudon - [#23](https://github.com/BlueBrain/emodel-generalisation/pull/23))

### General Changes

- 0.2.3 (Alexis Arnaudon - [#26](https://github.com/BlueBrain/emodel-generalisation/pull/26))

## [0.2.2](https://github.com/BlueBrain/emodel-generalisation/compare/0.2.1..0.2.2)

> 7 December 2023

### Fixes

- Load mod files after compile (Alexis Arnaudon - [#20](https://github.com/BlueBrain/emodel-generalisation/pull/20))

### General Changes

- 0.2.2 (Alexis Arnaudon - [#21](https://github.com/BlueBrain/emodel-generalisation/pull/21))

## [0.2.1](https://github.com/BlueBrain/emodel-generalisation/compare/0.2.0..0.2.1)

> 1 December 2023

### Chores And Housekeeping

- Relax deps versions (Alexis Arnaudon - [#18](https://github.com/BlueBrain/emodel-generalisation/pull/18))

### General Changes

- Release 0.2.1 (Alexis Arnaudon - [#19](https://github.com/BlueBrain/emodel-generalisation/pull/19))

## [0.2.0](https://github.com/BlueBrain/emodel-generalisation/compare/0.1.1..0.2.0)

> 30 November 2023

### New Features

- Add a nexus access point converter and improve CLI (Alexis Arnaudon - [#14](https://github.com/BlueBrain/emodel-generalisation/pull/14))
- Only Rin in cli (Alexis Arnaudon - [#16](https://github.com/BlueBrain/emodel-generalisation/pull/16))
- Add CLI (Alexis Arnaudon - [#10](https://github.com/BlueBrain/emodel-generalisation/pull/10))

### CI Improvements

- (dependabot) Bump actions/setup-node from 3 to 4 (Adrien Berchet - [#15](https://github.com/BlueBrain/emodel-generalisation/pull/15))

### General Changes

- Release 0.2.2 (Alexis Arnaudon - [#17](https://github.com/BlueBrain/emodel-generalisation/pull/17))
- Bump mikepenz/action-junit-report from 3 to 4 (dependabot[bot] - [#12](https://github.com/BlueBrain/emodel-generalisation/pull/12))
- Bump actions/checkout from 3 to 4 (Adrien Berchet - [#11](https://github.com/BlueBrain/emodel-generalisation/pull/11))

### Fixes

- Max xgboost version (arnaudon - [860a591](https://github.com/BlueBrain/emodel-generalisation/commit/860a591106febd4ea43b23c7fdac29b23964f96d))
- Licence (arnaudon - [c462664](https://github.com/BlueBrain/emodel-generalisation/commit/c4626640639f285259072198e841df59c453982b))

### General Changes

- Update README.md (Alexis Arnaudon - [682b7d0](https://github.com/BlueBrain/emodel-generalisation/commit/682b7d000c32e1181ff5719dbcae17c3579528a8))
- Add zenodo badge (Alexis Arnaudon - [084c57e](https://github.com/BlueBrain/emodel-generalisation/commit/084c57e46ac2ed4ef57dc4f5afbd3e534b8d0ac8))

## 0.1.1

> 21 August 2023

### New Features

- Update publish workflow (Alexis Arnaudon - [#9](https://github.com/BlueBrain/emodel-generalisation/pull/9))
- Improve readme (Werner Van Geit - [#4](https://github.com/BlueBrain/emodel-generalisation/pull/4))
- Added code (Alexis Arnaudon - [#1](https://github.com/BlueBrain/emodel-generalisation/pull/1))

### General Changes

- Fix layer str (Alexis Arnaudon - [#6](https://github.com/BlueBrain/emodel-generalisation/pull/6))
- Fix readme (Werner Van Geit - [#5](https://github.com/BlueBrain/emodel-generalisation/pull/5))

### General Changes

- Repository structure (arnaudon - [3992fcb](https://github.com/BlueBrain/emodel-generalisation/commit/3992fcba8e999c905cad7bb5b9b7301b54a74a1d))
- improve example readme (arnaudon - [7333517](https://github.com/BlueBrain/emodel-generalisation/commit/73335174bc3be6cee76c2b3a61f1740ca05b99fc))
- improve readme (arnaudon - [4ff2cad](https://github.com/BlueBrain/emodel-generalisation/commit/4ff2cad3a6876343f24250a72566bfa6172016bb))
- fix (arnaudon - [296f7d9](https://github.com/BlueBrain/emodel-generalisation/commit/296f7d985d4d0a99fdfd3d4e2e70a79c3dcf6ed4))
- use code (arnaudon - [f346812](https://github.com/BlueBrain/emodel-generalisation/commit/f346812a386e05a85ff568bfc4bf402d8e2d5dd2))
- move citation down (arnaudon - [12744da](https://github.com/BlueBrain/emodel-generalisation/commit/12744da4580d04a66b585afded40b024e3e1dd6d))
- fix lint (arnaudon - [420f2b8](https://github.com/BlueBrain/emodel-generalisation/commit/420f2b8501f330e73718ad0e29f6f5fdaca006d5))
- typo2 (arnaudon - [cc5d789](https://github.com/BlueBrain/emodel-generalisation/commit/cc5d789b98badcff3c7cd1385000a0a2db633972))
- typo (arnaudon - [ba360d7](https://github.com/BlueBrain/emodel-generalisation/commit/ba360d7289e42883b39b56c1609fe043728738b3))
