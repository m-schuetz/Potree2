
import {Gradients, Utils} from "potree";

class AttributesPanel{

	constructor(){
		this.element = document.createElement("div");

	}

	set(pointcloud){
		
		

	}

}

export function createAttributesPanel(){
	let elAttributes = document.createElement("div");
	elAttributes.id = "attributes_panel";

	let elTitle = document.createElement("div");
	elTitle.classList.add("subsection");
	elTitle.textContent = "Attributes";

	elAttributes.append(elTitle);

	{ // ATTRIBUTE SELECTION
		let elList = document.createElement("select");
		elList.multiple = true;
		elList.size = 5;

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

		elAttributes.appendChild(elList);
	}

	{ // GRADIENT
		const schemes = Object.keys(Gradients).map(name => ({name: name, values: Gradients[name]}));

		const elGradientSchemes = document.createElement("div");
		elGradientSchemes.id = "gradient_schemes";
		elGradientSchemes.style.display = "flex";

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

			elGradientSchemes.appendChild(elButton);
		}

		elAttributes.append(elGradientSchemes);
	}


	let panel = new AttributesPanel();
	panel.element = elAttributes;

	return panel;
}