
// see https://stackoverflow.com/a/31083391/913630

export class RangeSelect extends HTMLElement{
	constructor(){
		super();

		this.style.position = "relative";
		this.style.width = "100%";
		this.style.userSelect = "none";

		this.range = [0, 100];
		this.value = [10, 60];

		this.innerHTML = `
			<style>
				.abc{
					position: absolute; 
					pointer-events: none;
					// outline: none;
					-webkit-appearance: none;
					background: none; 
					width: 100%; 
					height: 0.5em;
					top: calc(50% - 0.25em);
				}

				input::-webkit-slider-thumb {
					pointer-events: all;
					position: relative;
					z-index: 1;
					 -webkit-appearance: none;
					width: 1.2em;
					height: 1.2em;
					border-radius: 1em;
					background-color: red;
				}

				input::-moz-range-thumb {
					pointer-events: all;
					position: relative;
					z-index: 10;
					-moz-appearance: none;
				}

				.background{
					box-sizing: border-box;
					margin: 2px;
					position: absolute;
					background: #eee;
					width: 100%;
					height: 0.4em;
					border-radius: 0.3em;
					border: 1px solid #aaa;
					top: calc(50% - 0.25em);
				}

				.selected{
					box-sizing: border-box;
					margin: 2px;
					position: absolute;
					background: red;
					width: 50%;
					height: 0.5em;
					border-radius: 0.5em;
					top: calc(50% - 0.25em);
				}

				.marker{
					position: absolute;
					left: 100px;
					top: 100px;
					border: 1px solid black;
					background: white;
					padding: 0px 3px;
					font-family: calibri;
					font-weight: bold;
					font-size: 0.75em;
					z-index: 100;
					color: black;
				}

				.test{
					position: relative;
				}

			</style>
			<span class="background"></span>
			<span class="selected"></span>
			<span class="marker" id="marker_min">abc</span>
			<span class="marker" id="marker_max">abc</span>
			<input type="range" min="${this.range[0]}" max="${this.range[1]}" value="${this.value[0]}" step="0.01" class="abc" />
			<input type="range" min="${this.range[0]}" max="${this.range[1]}" value="${this.value[1]}" step="0.01" class="abc" />

		`;

		this.elSliders = this.querySelectorAll("input");
		this.elBackground = this.querySelector(`.background`);
		this.elSelected = this.querySelector(`.selected`);
		this.elMin = this.querySelector(`#marker_min`);
		this.elMax = this.querySelector(`#marker_max`);

		this.elSliders[0].addEventListener("input", this.onInput.bind(this));
		this.elSliders[1].addEventListener("input", this.onInput.bind(this));

		this.onInput();
	}

	setRange(min, max){
		this.range = [min, max];

		this.elSliders[0].min = min;
		this.elSliders[0].max = max;
		this.elSliders[1].min = min;
		this.elSliders[1].max = max;
	}

	setValue(min, max){
		this.value = [min, max];

		this.elSliders[0].value = min;
		this.elSliders[1].value = max;

		this.onInput();
	}

	onInput(event){
		
		let values = [this.elSliders[0].value, this.elSliders[1].value];
		let min = Math.min(...values);
		let max = Math.max(...values);

		let rangeSize = this.range[1] - this.range[0];
		let u_min = (min - this.range[0]) / rangeSize;
		let u_max = (max - this.range[0]) / rangeSize;

		this.elSelected.style.left = `${100 * u_min}%`;
		this.elSelected.style.width = `${100 * (u_max - u_min)}%`;

		{
			let selectionBox = this.elSelected.getBoundingClientRect();

			this.elMin.innerText = min.toLocaleString();
			let elMinBox = this.elMin.getBoundingClientRect();

			// this.elMin.style.left = `calc(${selectionBox.left - 0.5 * elMinBox.width}px)`;
			// this.elMin.style.top = `calc(${selectionBox.top}px - 2em)`;

			let elSlidersBox = this.elSliders[0].getBoundingClientRect();
			this.elMin.style.left = `calc(${elSlidersBox.width * u_min - elMinBox.width / 2}px)`;
			this.elMin.style.top = `0px`;

			this.elMax.innerText = max.toLocaleString();
			let elMaxBox = this.elMax.getBoundingClientRect();
			// this.elMax.style.left = `calc(${selectionBox.right - 0.5 * elMaxBox.width}px)`;
			// this.elMax.style.top = `calc(${selectionBox.top}px - 2em)`;
			this.elMax.style.left = `calc(${elSlidersBox.width * u_max - elMaxBox.width / 2}px)`;
			this.elMax.style.top = `0px`;
		}

		this.value = [min, max];

		if(event){
			event.stopPropagation();
		}

		this.dispatchEvent(new InputEvent("input"));

		return false;
	}
}

customElements.define('range-select', RangeSelect);