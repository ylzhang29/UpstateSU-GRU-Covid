---
layout: default
---

<div class="viewof-view"></div>

<script type="module">
  import {
    Runtime,
    Inspector,
  } from "https://cdn.jsdelivr.net/npm/@observablehq/runtime@4/dist/runtime.js";
  import define from "https://api.observablehq.com/d/91c845853e2a1ef0.js?v=3";
  const main = new Runtime().module(define, (name) => {
    if (name === "viewof view") return Inspector.into(".viewof-view")();
  });

  main.redefine('url', 'https://raw.githubusercontent.com/ylzhang29/UpstateSU-GRU-Covid/main/Forecast/new_forecast.csv')
</script>
