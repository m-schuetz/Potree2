
import {Gradients, Utils, Potree} from "potree";

class Panel{

	constructor(){
		this.element = document.createElement("div");
		this.pointcloud = null;

		this.element.id = "attributes_panel";
		this.element.innerHTML = `
		
			<div class="subsection">Attributes</div>

			<select id="attributes_list" >
			</select>

			<div id="gradient_schemes" style="display: flex">
			</div>
		`;

		this.elAttributeList = this.element.querySelector("#attributes_list");
		this.elGradientSchemes = this.element.querySelector("#gradient_schemes");

		this.elAttributeList.onchange = () => {
			this.onAttributeSelected();
		};

		this.updateGradientSchemes();
	}

	onAttributeSelected(){
		// console.log(this.elAttributeList.value);

		Potree.settings.attribute = this.elAttributeList.value;
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

		let statsList = this.pointcloud.root.geometry.statsList;

		let weighted = [];
		for(let i = 0; i < statsList.length; i++){
			let stats = statsList[i];
			let index = preferredOrder.indexOf(stats.name);

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