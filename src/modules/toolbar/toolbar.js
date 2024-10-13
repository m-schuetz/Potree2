
import {Gradients, Utils} from "potree";
import {PointMeasure, DistanceMeasure, HeightMeasure} from "potree";

export async function installToolbar(element, potree){

	let {css} = await import("./toolbar.css.js");
	
	let dir = new URL(import.meta.url + "/../").href;

	let style = document.createElement('style');
	style.innerHTML = css;
	document.getElementsByTagName('head')[0].appendChild(style);

	let elToolbar = document.createElement("div");
	elToolbar.id = "potree_toolbar";

	elToolbar.innerHTML = `
		<span>
			<div class="potree_toolbar_label">
				Attribute
			</div>
			<div id="attributes_panel">

			</div>
		</span>

		<span class="potree_toolbar_separator"></span>

		<span>
			<div class="potree_toolbar_label">
				Gradients
			</div>
			<div>
				<span id="gradient_schemes" name="gradient_schemes"></span>
			</div>
			<div id="gradient_drop">
				
			</div>
		</span>

		<span class="potree_toolbar_separator"></span>

		<span>
			<div class="potree_toolbar_label">
				Measure
			</div>
			<div id="measures_panel"></div>
		</span>

		<span class="potree_toolbar_separator"></span>

		<span>
			<div class="potree_toolbar_label">
				Display
			</div>
			<div id="display_panel"></div>
		</span>

		<!--
		<span class="potree_toolbar_separator"></span>

		<span>
			<div class="potree_toolbar_label">
				<span data-i18n="appearance.nb_max_pts">Point Budget</span>: 
				<span id="lblPointBudget" style="display: inline-block; width: 3em;">3.0M</span>
			</div>
			<div>
				<input id="sldPointBudget" type="range" min="1" max="100" value="30" class="slider">
			</div>
		</span>
		-->
	`;

	{ // ATTRIBUTE
		const elContainer = elToolbar.querySelector("#attributes_panel");

		{
			let elList = document.createElement("select");
			elList.multiple = true;
			elList.size = 3;

			let items = [
				"rgba", 
				"elevation", 
				"intensity", 
				"classification", 
				"number of returns", 
				"gps-time"
			];

			for(let item of items){
				let elOption = document.createElement("option");
				elOption.innerText = item;
				elOption.value = item;

				elList.appendChild(elOption);
			}

			elContainer.appendChild(elList);

			elList.onchange = () => {
				Potree.settings.attribute = elList.value;
			};

			let elRgba = elContainer.querySelector(`option[value="rgba"]`);
			elRgba.selected = true;
		}

	}

	{ // MEASURE
		const elMeasures = elToolbar.querySelector("#measures_panel");
		
		{ // POINT
			let elButton = document.createElement("input");
			elButton.classList.add("potree_toolbar_button");
			elButton.type = "button";
			elButton.title = "Point Measure";
			elButton.style.backgroundImage = `url(${dir}/icons/point.svg)`;

			elButton.addEventListener("click", () => {
				potree.measure.startMeasuring(new PointMeasure());
			});
			
			elMeasures.appendChild(elButton);
		}

		{ // DISTANCE
			let elButton = document.createElement("input");
			elButton.classList.add("potree_toolbar_button");
			elButton.type = "button";
			elButton.title = "Distance Measure";
			elButton.style.backgroundImage = `url(${dir}/icons/distance.svg)`;

			elButton.addEventListener("click", () => {
				potree.measure.startMeasuring(new DistanceMeasure());
			});
			
			elMeasures.appendChild(elButton);
		}

		{ // CIRCLE
			let elButton = document.createElement("input");
			elButton.classList.add("potree_toolbar_button");
			elButton.type = "button";
			elButton.title = "Circle Measure";
			elButton.style.backgroundImage = `url(${dir}/icons/circle.svg)`;
			
			elMeasures.appendChild(elButton);
		}

	}

	{ // GRADIENT
		const schemes = Object.keys(Gradients).map(name => ({name: name, values: Gradients[name]}));

		const elGradientSchemes = elToolbar.querySelector("#gradient_schemes");

		for(const scheme of schemes){
			let elButton = document.createElement("input");
			elButton.type = "button";

			let stops = [];
			let n = 16;
			for(let i = 0; i <= n; i++){
				let u = 1 - i / n;
				let stopVal = scheme.values.get(u);

				let [r, g, b, a] = stopVal.map(v => parseInt(v));
				let percent = (1 - u) * 100;

				let stopString = `rgba(${r}, ${g}, ${b}) ${percent}%`;
				
				stops.push(stopString);
			}

			let stopsString = stops.join(", ");
			let cssGradient = `linear-gradient(to bottom, ${stopsString})`;

			elButton.style.backgroundImage = cssGradient;
			elButton.title = scheme.name;

			elButton.addEventListener("click", () => {
				Potree.settings.gradient = scheme.values;
				// Potree.settings.attribute = "elevation";
			});


			elButton.classList.add("potree_toolbar_gradient_button");

			elGradientSchemes.appendChild(elButton);
		}

		// { // DROPDOWN

		// 	let elDropdown = elToolbar.querySelector("#gradient_drop");

		// 	let elButton = document.createElement("input");
		// 	elButton.type = "button";
		// 	// elButton.value = "···";
		// 	elButton.classList.add("potree_toolbar_dropdown_button");
		// 	elButton.style.backgroundImage = `url(${dir}/icons/dotdotdot.svg)`;

		// 	// elButton.innerHTML = `
			
		// 	// <svg height="32" width="32">
		// 	// 	<text x="0" y="15" fill="red">I love SVG!</text>
		// 	// </svg>

		// 	// `;

		// 	elDropdown.appendChild(elButton);

		// }
	}

	// { // POINT BUDGET

	// 	let slider = elToolbar.querySelector("#sldPointBudget");
	// 	let lblPointBudget = elToolbar.querySelector("#lblPointBudget");

	// 	let updateLabel = () => {
	// 		let budget = Number(slider.value) * 100_000;
	// 		let stringValue = (budget / 1_000_000).toFixed(1);
	// 		let string = stringValue.toLocaleString() + "M";

	// 		lblPointBudget.innerHTML = string;

	// 		Potree.settings.pointBudget = budget;
	// 	};

	// 	slider.oninput = updateLabel;

	// 	updateLabel();

	// }

	element.parentElement.appendChild(elToolbar);



}


