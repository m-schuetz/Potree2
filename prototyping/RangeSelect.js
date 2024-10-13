
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
			</style>
			<span class="background"></span>
			<span class="selected"></span>
			<input type="range" min="${this.range[0]}" max="${this.range[1]}" value="${this.value[0]}" class="abc" />
			<input type="range" min="${this.range[0]}" max="${this.range[1]}" value="${this.value[1]}" class="abc" />

		`;

		this.elSliders = this.querySelectorAll("input");
		this.elBackground = this.querySelector(`.background`);
		this.elSelected = this.querySelector(`.selected`);

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

		this.value = [min, max];

		if(event){
			event.stopPropagation();
		}

		this.dispatchEvent(new InputEvent("input"));

		return false;
	}
}

customElements.define('range-select', RangeSelect);