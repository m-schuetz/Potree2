
import {Gradients, Utils, Potree, Attribute_Custom} from "potree";

function toHexString(color){

	let hR = parseInt(255 * color[0]).toString(16).padStart(2, 0);
	let hG = parseInt(255 * color[1]).toString(16).padStart(2, 0);
	let hB = parseInt(255 * color[2]).toString(16).padStart(2, 0);

	let str = `#${hR}${hG}${hB}`;

	return str;
}

class Panel{

	constructor(){
		this.element = document.createElement("div");
		this.pointcloud = null;

		this.element.id = "attributes_panel";
		this.element.innerHTML = `

			<style>
				.subsubsection{
					padding: 15px 0px;
				}
			</style>
		
			<div class="subsection">Attributes</div>

			<select id="attributes_list"></select>

			<div class="subsubsection" id="gradient_schemes" style="display: flex"></div>

			<div class="subsubsection" id="attributes_gamma_brightness_contrast" style="display: flex"></div>

			<div class="subsubsection" id="attributes_scalar" style="display: flex"></div>

			<div class="subsubsection" id="attributes_listing" style="display: grid; grid-template-columns: 3em 1fr 3em"></div>

		`;

		this.elAttributeList = this.element.querySelector("#attributes_list");
		this.elGradientSchemes = this.element.querySelector("#gradient_schemes");
		this.elGammaBrightnessContrast = this.element.querySelector("#attributes_gamma_brightness_contrast");
		this.elScalar = this.element.querySelector("#attributes_scalar");
		this.elListing = this.element.querySelector("#attributes_listing");

		this.elAttributeList.onchange = () => {
			this.onAttributeSelected();
		};

		this.updateSettings();
	}

	onAttributeSelected(){
		Potree.settings.attribute = this.elAttributeList.value;

		this.updateSettings();
	}

	updateSettings(){

		let elGammaBrightnessContrast = this.elGammaBrightnessContrast;
		let elScalar = this.elScalar;
		let elGradient = this.elGradientSchemes;
		let elListing = this.elListing;

		let show = (...args) => {
			elGammaBrightnessContrast.style.display = args.includes(elGammaBrightnessContrast) ? "flex" : "none";
			elScalar.style.display = args.includes(elScalar) ? "flex" : "none";
			elGradient.style.display = args.includes(elGradient) ? "flex" : "none";
			elListing.style.display = args.includes(elListing) ? "grid" : "none";

			if(args.includes(elGammaBrightnessContrast)){
				this.updateGammaBrightnessContrast();
			}

			if(args.includes(elScalar)){
				this.updateScalar();
			}
			
			if(args.includes(elListing)){
				this.updateListing();
			}

			if(args.includes(elGradient)){
				this.updateGradientSchemes();
			}
		};

		if(!this.pointcloud){
			show();
			return;
		}

		let attributeName = Potree.settings.attribute;
		let settings = this.pointcloud.material.attributes.get(attributeName);

		if(!settings){
			return;
		}else if(settings.constructor.name === "Attribute_RGB"){
			show();
		}else if(settings.constructor.name === "Attribute_Scalar"){
			show(elScalar, elGradient);
		}else if(settings.constructor.name === "Attribute_Listing"){
			show(elListing);
		}else{
			show();
		}

	}

	updateGammaBrightnessContrast(){
		let element = this.elGammaBrightnessContrast;

		elScalar.style.display = "block";

		// elScalar.innerHTML = `
		
		// 	<div style="display: grid; grid-template-columns: 4em 1fr; gap: 5px 10px;">

		// 		<span>Gamma</span>
		// 		<range-select id="sldGamma"></range-select>

		// 		<span>Brightness</span>
		// 		<range-select id="sldBrightness"></range-select>

		// 		<span>Contrast</span>
		// 		<range-select id="sldContrast"></range-select>

		// 	</div>

		// `;

		// let elGamma = elScalar.querySelector("#sldGamma");
		// let elBrightness = elScalar.querySelector("#sldBrightness");
		// let elContrast = elScalar.querySelector("#sldContrast");

		// let attributeName = Potree.settings.attribute;
		// let settings = this.pointcloud.material.attributes.get(attributeName);

		// if(settings){

		// 	elGamma.setRange(0, 4);
		// 	elBrightness.setRange(-1, 1);
		// 	elContrast.setRange(-1, 1);

		// 	elRange.setRange(...settings.range);
		// 	elRange.setValue(settings.stats.min, settings.stats.max);

		// 	elRange.addEventListener("input", () => {
		// 		// console.log(elRange.value);
		// 		settings.range = elRange.value;
		// 	});
		// }
	}

	updateScalar(){
		let elScalar = this.elScalar;

		elScalar.style.display = "block";

		elScalar.innerHTML = `
		
			<div style="display: grid; grid-template-columns: 5em 1fr; gap: 2px 2px;">

				<span style="height: 3em">Range</span>
				<range-select id="sldScalarRange"></range-select>

				<span style="height: 3em">Clamp</span>
				<input type="checkbox" id="chkClamp"></input>

			</div>

		`;

		let elRange = elScalar.querySelector("#sldScalarRange");
		let elClamp = elScalar.querySelector("#chkClamp");

		let attributeName = Potree.settings.attribute; 
		let settings = this.pointcloud.material.attributes.get(attributeName);

		if(settings){
			elRange.setRange(...settings.range);
			elRange.setValue(settings.stats.min, settings.stats.max);

			elRange.addEventListener("input", () => {
				settings.range = elRange.value;
			});

			elClamp.checked = settings.clamp;
			elClamp.addEventListener("change", () => {
				settings.clamp = elClamp.checked;
			});
		}

	}

	updateListing(){
		let elListing = this.elListing;
		elListing.innerHTML = "";

		let attributeName = Potree.settings.attribute;
		let setting = this.pointcloud.material.attributes.get(attributeName);

		if(!setting){
			return;
		}

		for(let key of Object.keys(setting.listing)){

			let value = setting.listing[key];

			let elIndex = document.createElement("span");
			let elLabel = document.createElement("span");
			let elColorPicker = document.createElement("input");

			let displayKey = key.replace("DEFAULT", "-");

			elIndex.innerText = `${displayKey}`;

			elLabel.innerText = `${value.name}`;
			elLabel.style.whiteSpace = "nowrap";

			elColorPicker.type = "color";
			elColorPicker.style.width = "3em";
			elColorPicker.value = toHexString(value.color);
			elColorPicker.addEventListener("input", () => {
				let str = elColorPicker.value;

				let r = Number(`0x${str.slice(1, 3)}`);
				let g = Number(`0x${str.slice(3, 5)}`);
				let b = Number(`0x${str.slice(5, 7)}`);

				value.color[0] = 255 * r;
				value.color[1] = 255 * g;
				value.color[2] = 255 * b;

			});
			

			elListing.append(elIndex, elLabel, elColorPicker);

		}

	}

	updateGradientSchemes(){
		const schemes = Object.keys(Gradients).map(name => ({name: name, values: Gradients[name]}));

		let elGrid = document.createElement("span");
		elGrid.style.display = "grid";
		elGrid.style.gridTemplateColumns = "5em 1fr";
		elGrid.style.width = "100%";

		elGrid.innerHTML = `
			<span>Gradients</span>
			<span id="gradients" style="display: flex"></span>
		`;

		let elGradients = elGrid.querySelector("#gradients");

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
			});


			elButton.classList.add("potree_gradient_button");

			elGradients.append(elButton);
		}

		this.elGradientSchemes.innerHTML = "";
		this.elGradientSchemes.append(elGrid);
	}

	updateAttributesList(){

		let preferredOrder = [
			// "rgb", "rgba", "RGB", "RGBA",
			// "intensity",
			// "classification",
			// "gps-time",
		];

		let blacklist = [];
		// let blacklist = ["XYZ", "position"];

		let attributes = this.pointcloud.material.attributes;

		let weighted = [];
		let i = 0;
		for(let [name, attribute] of attributes){
			let index = preferredOrder.indexOf(name);

			if(blacklist.includes(name)){
				continue;
			}

			let weight = index >= 0 ? index : 100 + i;

			weighted.push({name, weight, attribute});
			i++;
		}

		// weighted.push({name: "elevation", weight: 4});
		weighted.sort( (a, b) => {
			return a.weight - b.weight;
		});

		this.elAttributeList.innerHTML = "";

		let customGroupStarted = false;

		let elOptgroupStandard = document.createElement("optgroup");
		elOptgroupStandard.label = "standard attributes";
		let elOptgroupCustom = document.createElement("optgroup");
		elOptgroupCustom.label = "extended attributes";

		for(let item of weighted){

			let name = item.name;
			let elOption = document.createElement("option");
			elOption.innerText = name;
			elOption.value = name;

			if(item.attribute.extended){
				elOptgroupCustom.appendChild(elOption);
			}else{
				elOptgroupStandard.appendChild(elOption);
			}

			// this.elAttributeList.appendChild(elOption);
		}

		this.elAttributeList.appendChild(elOptgroupStandard);
		this.elAttributeList.appendChild(elOptgroupCustom);

		this.elAttributeList.size = weighted.length + 3;
		this.elAttributeList.value = Potree.settings.attribute;
	}

	set(pointcloud){

		this.pointcloud = pointcloud;

		let onChange = () => {
			this.updateAttributesList();
			this.updateSettings();
		};
		this.pointcloud.events.onMaterialChanged(onChange);

		onChange();
		
		// if(pointcloud.root?.geometry?.statsList){
		// 	this.updateAttributesList();
		// 	this.updateSettings();
		// }else{
		// 	let onRootNodeLoaded = (event) => {
		// 		this.updateAttributesList();
		// 		this.updateSettings();
		// 	};
		// 	onRootNodeLoaded.isOneTimeEvent = true;
		// 	pointcloud.events.onRootNodeLoaded(onRootNodeLoaded);
		// }

	}

}

export function createAttributesPanel(){

	let panel = new Panel();

	Potree.events.onPointcloudLoaded((pointcloud) => {
		panel.set(pointcloud);
	});

	return panel;
}