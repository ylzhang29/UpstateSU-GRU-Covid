import { Inspector } from "https://cdn.jsdelivr.net/npm/@observablehq/runtime@4/dist/runtime.js";

export default class MapInspector extends Inspector {
  constructor(selector) {
    let el;
    super((el = document.querySelector(selector)));
    this.el = el;
  }

  pending() {
    super.pending();
    this.el.innerText = "Loading... This may take a few seconds...";
  }

  fulfilled(value) {
    this.el.innerText = "";
    super.fulfilled(value);
  }
}
