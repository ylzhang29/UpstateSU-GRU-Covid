---
layout: default
---

<div class="viewof-view"></div>

<script type="module">
import {Runtime, Library} from "https://cdn.jsdelivr.net/npm/@observablehq/runtime@4/dist/runtime.js";
import define from "https://api.observablehq.com/d/447375f0c38a052b.js?v=3";
import Inspector from './assets/js/LoadableInspector.js';

const stdlib = new Library()

const container = document.querySelector('#main_content')
const width = stdlib.Generators.observe(c => {
  const handleResize = () => c(container.offsetWidth)
  window.addEventListener('resize', handleResize)
  c(container.offsetWidth)
  return () => window.removeEventListener('resize', handleResize)
})

const runtime = new Runtime(Object.assign(stdlib, { width }))

const main = runtime.module(define, name => {
  if (name === "viewof view") return new Inspector(".viewof-view");
});
</script>
