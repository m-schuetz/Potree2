
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
			<div>
				<img name="action_elevation" src="${dir}/icons/profile.svg" class="annotation-action-icon" style="width: 2em; height: auto;"/>
				<img name="action_rgb" src="${dir}/icons/rgb.svg" class="annotation-action-icon" style="width: 2em; height: auto;"/>
			</div>
		</span>

		<span class="potree_toolbar_separator"></span>

		<span>
			<div class="potree_toolbar_label">
				Gradient
			</div>
			<div>
				<span name="gradient_schemes"></span>
			</div>
		</span>

		<span class="potree_toolbar_separator"></span>

		<span>
			<div class="potree_toolbar_label">
				Measure
			</div>
			<div>
				<img name="action_measure_point" src="${dir}/icons/point.svg" class="annotation-action-icon" style="width: 2em; height: auto;"/>
				<img name="action_measure_distance" src="${dir}/icons/distance.svg" class="annotation-action-icon" style="width: 2em; height: auto;"/>
				<img name="action_measure_circle" src="${dir}/icons/circle.svg" class="annotation-action-icon" style="width: 2em; height: auto;"/>
			</div>
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

	{ // GRADIENT
		const schemes = Object.keys(Gradients).map(name => ({name: name, values: Gradients[name]}));
		// const elGradientSchemes = elToolbar.find("span[name=gradient_schemes]");

		const elGradientSchemes = Utils.domFindByName(elToolbar, "gradient_schemes");

		for(const scheme of schemes){
			const elButton = document.createElement("span");

			const svg = Utils.createSvgGradient(scheme.values);
			svg.setAttributeNS(null, "class", `button-icon`);
			svg.style.height = "2em";
			svg.style.width = "1.3em";

			elButton.appendChild(svg);

			// elButton.click( () => {
			// 	for(const pointcloud of viewer.scene.pointclouds){
			// 		pointcloud.material.activeAttributeName = "elevation";
			// 		pointcloud.material.gradient = Potree.Gradients[scheme.name];
			// 	}
			// });

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


