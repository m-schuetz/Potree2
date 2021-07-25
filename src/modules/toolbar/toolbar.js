
import {Gradients, Utils} from "potree";

export async function installToolbar(element){

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
			<div id="attributes_panel"></div>
		</span>

		<span class="potree_toolbar_separator"></span>

		<span>
			<div class="potree_toolbar_label">
				Gradients
			</div>
			<div>
				<span id="gradient_schemes" name="gradient_schemes"></span>
			</div>
		</span>

		<span class="potree_toolbar_separator"></span>

		<span>
			<div class="potree_toolbar_label">
				Measure
			</div>
			<div id="measures_panel"></div>
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
		
		{ // ELEVATION
			let elButton = document.createElement("input");
			elButton.classList.add("potree_toolbar_button");
			elButton.type = "button";
			elButton.style.backgroundImage = `url(${dir}/icons/profile.svg)`;

			elButton.addEventListener("click", () => {
				Potree.settings.dbgAttribute = "elevation";
			});
			
			elContainer.appendChild(elButton);
		}

		{ // RGB
			let elButton = document.createElement("input");
			elButton.classList.add("potree_toolbar_button");
			elButton.type = "button";
			elButton.style.backgroundImage = `url(${dir}/icons/rgb.svg)`;

			elButton.addEventListener("click", () => {
				Potree.settings.dbgAttribute = "rgba";
			});
			
			elContainer.appendChild(elButton);
		}

	}

	{ // MEASURE
		const elMeasures = elToolbar.querySelector("#measures_panel");
		
		{ // POINT
			let elButton = document.createElement("input");
			elButton.classList.add("potree_toolbar_button");
			elButton.type = "button";
			elButton.style.backgroundImage = `url(${dir}/icons/point.svg)`;
			
			elMeasures.appendChild(elButton);
		}

		{ // DISTANCE
			let elButton = document.createElement("input");
			elButton.classList.add("potree_toolbar_button");
			elButton.type = "button";
			elButton.style.backgroundImage = `url(${dir}/icons/distance.svg)`;
			
			elMeasures.appendChild(elButton);
		}

		{ // CIRCLE
			let elButton = document.createElement("input");
			elButton.classList.add("potree_toolbar_button");
			elButton.type = "button";
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
				let u = i / n;
				let stopVal = scheme.values.get(u);

				let [r, g, b, a] = stopVal.map(v => parseInt(v));
				let percent = u * 100;

				let stopString = `rgba(${r}, ${g}, ${b}) ${percent}%`;
				
				stops.push(stopString);
			}

			let stopsString = stops.join(", ");
			let cssGradient = `linear-gradient(to bottom, ${stopsString})`;

			elButton.style.backgroundImage = cssGradient;

			elButton.addEventListener("click", () => {
				Potree.settings.gradient = scheme.values;
				Potree.settings.dbgAttribute = "elevation";
			});


			elButton.classList.add("potree_toolbar_gradient_button");

			elGradientSchemes.appendChild(elButton);
		}
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


