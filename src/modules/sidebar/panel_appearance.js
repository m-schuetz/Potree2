
import {Gradients, Utils} from "potree";
let dir = new URL(import.meta.url + "/../").href;

class Panel{

	constructor(){
		this.element = document.createElement("div");

	}

	set(pointcloud){
		// connect attributes
	}

}

export function createPanel(){
	let elPanel = document.createElement("div");
	elPanel.id = "appearance_panel";
	elPanel.classList.add("subsection_panel");

	let elTitle = document.createElement("div");
	elTitle.classList.add("subsection");
	elTitle.textContent = "Appearance";

	elPanel.append(elTitle);

	{ // Point Budget

		let elContainer = document.createElement("div");
		elContainer.style.display = "grid";
		elContainer.style.gridTemplateColumns = "1fr 2fr 4em";
		elContainer.style.gridGap = "5px 10px";

		elPanel.append(elContainer);

		let addSlider = (args) => {

			let [min, max] = args.range;

			let template = document.createElement('template');
			template.innerHTML = `
				<sidebarlabel>${args.label}</sidebarlabel>
				<input type="range" min="${min}" max="${max}" value="${args.value}" style="width: 100%" name=${args.elementName}>
				<sidebarlabel>abc M</sidebarlabel>
			`;
			let nodes = template.content.childNodes;
			elContainer.append(...nodes);

			let elSlider = elContainer.querySelector(`input[name=${args.elementName}]`);
			let elValue = elContainer.querySelectorAll(`sidebarlabel`)[1];

			elSlider.addEventListener("input", () => {
				args.onChange(elSlider, elValue);
			});
			args.onChange(elSlider, elValue);
		};

		let addCheckbox = (label, elementName, initialValue, onChange) => {
			let template = document.createElement('template');
			template.innerHTML = `
				<sidebarlabel>${label}</sidebarlabel>
				<input type="checkbox" name="${elementName}">
				<sidebarlabel></sidebarlabel>
			`;
			let nodes = template.content.childNodes;
			elContainer.append(...nodes);

			let elCheckbox = elContainer.querySelector(`input[name=${elementName}]`);
			elCheckbox.checked = initialValue;
			elCheckbox.addEventListener("change", () => {
				onChange(elCheckbox);
			});
		};


		addSlider({
			label: "Point Budget", 
			elementName: "sldPointBudget",
			range: [1_000_000, 5_000_000], 
			value: Potree.settings.pointBudget,
			onChange: (elSlider, elValue) => {
				Potree.settings.pointBudget = Number(elSlider.value);

				let str = (Number(elSlider.value) / 1_000_000).toFixed(1) + " M";
				elValue.innerText = str;
			},
		});

		addCheckbox("dilate", "chkDilate", Potree.settings.dilateEnabled, 
			(checkbox) => {Potree.settings.dilateEnabled = checkbox.checked;}
		);

		addCheckbox("Eye-Dome-Lighting", "chkEDL", Potree.settings.edlEnabled, 
			(checkbox) => {Potree.settings.edlEnabled = checkbox.checked;}
		);

		addCheckbox("High-Quality", "chkHQS", Potree.settings.hqsEnabled, 
			(checkbox) => {Potree.settings.hqsEnabled = checkbox.checked;}
		);

		addCheckbox("show bounding box", "chkShowBoundingBox", Potree.settings.showBoundingBox, 
			(checkbox) => {Potree.settings.showBoundingBox = checkbox.checked;}
		);

		addCheckbox("update", "chkUpdate", Potree.settings.updateEnabled, 
			(checkbox) => {Potree.settings.updateEnabled = checkbox.checked;}
		);


	}

	


	let panel = new Panel();
	panel.element = elPanel;

	return panel;
}