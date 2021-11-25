
import {Gradients, Utils, Potree} from "potree";

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
		
			<div class="subsection">Attributes</div>

			<select id="attributes_list"></select>

			<div id="gradient_schemes" style="display: flex"></div>

			<div id="attributes_gamma_brightness_contrast" style="display: flex"></div>

			<div id="attributes_scalar" style="display: flex">
				abc
			</div>

			<div id="attributes_listing" style="display: grid; grid-template-columns: 3em 1fr 3em">
				abc
			</div>

		`;

		this.elAttributeList = this.element.querySelector("#attributes_list");
		this.elGradientSchemes = this.element.querySelector("#gradient_schemes");
		this.elGammeBrightnessContrast = this.element.querySelector("#attributes_gamma_brightness_contrast");
		this.elScalar = this.element.querySelector("#attributes_scalar");
		this.elListing = this.element.querySelector("#attributes_listing");

		this.elAttributeList.onchange = () => {
			this.onAttributeSelected();
		};

		this.updateGradientSchemes();
	}

	onAttributeSelected(){
		Potree.settings.attribute = this.elAttributeList.value;

		this.updateSettings();
	}

	updateSettings(){

		let attributeName = Potree.settings.attribute;
		let settings = this.pointcloud.material.attributes.get(attributeName);

		let elScalar = this.elScalar;
		let elGradient = this.elGradientSchemes;
		let elListing = this.elListing;

		let show = (...args) => {
			elScalar.style.display = args.includes(elScalar) ? "flex" : "none";
			elGradient.style.display = args.includes(elGradient) ? "flex" : "none";
			elListing.style.display = args.includes(elListing) ? "grid" : "none";

			if(args.includes(elScalar)){
				this.updateScalar();
			}
			
			if(args.includes(elListing)){
				this.updateListing();
			}
		};

		if(settings.constructor.name === "Attribute_RGB"){
			show();
		}else if(settings.constructor.name === "Attribute_Scalar"){
			show(elScalar, elGradient);
		}else if(settings.constructor.name === "Attribute_Listing"){
			show(elListing);
		}else{
			show();
		}

	}

	updateScalar(){
		// this.elScalar.innerHTML


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

			

			elListing.append(elIndex, elLabel, elColorPicker);

		}

	}

	updateGradientSchemes(){
		const schemes = Object.keys(Gradients).map(name => ({name: name, values: Gradients[name]}));

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

			this.elGradientSchemes.appendChild(elButton);
		}
	}

	updateAttributesList(){

		let preferredOrder = [
			"rgb", "rgba", "RGB", "RGBA",
			"intensity",
			"classification",
			"gps-time",
		];

		let blacklist = ["XYZ", "position"];

		let statsList = this.pointcloud.root.geometry.statsList;

		let weighted = [];
		for(let i = 0; i < statsList.length; i++){
			let stats = statsList[i];
			let index = preferredOrder.indexOf(stats.name);

			if(blacklist.includes(stats.name)){
				continue;
			}

			let weight = index >= 0 ? index : 100 + i;

			weighted.push({name: stats.name, weight: weight});
		}
		weighted.push({name: "elevation", weight: 4});
		weighted.sort( (a, b) => {
			return a.weight - b.weight;
		});

		for(let item of weighted){
			let name = item.name;
			let elOption = document.createElement("option");
			elOption.innerText = name;
			elOption.value = name;

			this.elAttributeList.appendChild(elOption);
		}

		this.elAttributeList.size = weighted.length;
		this.elAttributeList.value = Potree.settings.attribute;


	}

	

	set(pointcloud){

		this.pointcloud = pointcloud;
		
		if(pointcloud.root?.geometry?.statsList){
			this.updateAttributesList();
		}else{
			let onRootNodeLoaded = (event) => {
				this.updateAttributesList();
			};
			onRootNodeLoaded.isOneTimeEvent = true;
			Potree.events.onRootNodeLoaded(onRootNodeLoaded);
		}

	}

}

export function createAttributesPanel(){

	let panel = new Panel();

	Potree.events.onPointcloudLoaded((pointcloud) => {
		panel.set(pointcloud);
	});

	return panel;
}