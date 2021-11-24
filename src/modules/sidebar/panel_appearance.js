
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

	let elTitle = document.createElement("div");
	elTitle.classList.add("subsection");
	elTitle.textContent = "Appearance";

	elPanel.append(elTitle);

	{ // Point Budget

		let elContainer = document.createElement("div");
		elContainer.style.display = "grid";
		elContainer.style.gridTemplateColumns = "1fr 2fr 4em";
		elContainer.style.gridGap = "5px 10px";

		elContainer.innerHTML = `
			<sidebarlabel>Point Budget</sidebarlabel>
			<input type="range" style="width: 100%">
			<sidebarlabel>1.4 M</sidebarlabel>

			<sidebarlabel>Point Size</sidebarlabel>
			<input type="range" style="width: 100%">
			<sidebarlabel>5 px</sidebarlabel>

			<sidebarlabel>Min Node Size</sidebarlabel>
			<input type="range" style="width: 100%">
			<sidebarlabel>56543 px</sidebarlabel>

			<sidebarlabel>dilate</sidebarlabel>
			<input type="checkbox" name="chkDilate">
			<sidebarlabel></sidebarlabel>

			<sidebarlabel>Eye-Dome-Lighting</sidebarlabel>
			<input type="checkbox" name="chkEDL">
			<sidebarlabel></sidebarlabel>

			<sidebarlabel>High Quality</sidebarlabel>
			<input type="checkbox" name="chkHQS">
			<sidebarlabel></sidebarlabel>

			<sidebarlabel>show bounding box</sidebarlabel>
			<input type="checkbox" name="chkBoundingBox">
			<sidebarlabel></sidebarlabel>

			<sidebarlabel>update</sidebarlabel>
			<input type="checkbox" name="chkUpdate">
			<sidebarlabel></sidebarlabel>

		`;

		elPanel.append(elContainer);

		let hookCheckbox = (sliderName, initialValue, onChange) => {
			let elCheckbox = elPanel.querySelector(`input[name=${sliderName}]`);
			elCheckbox.checked = initialValue;

			elCheckbox.addEventListener("change", () => {
				onChange(elCheckbox);
			});
		}

		hookCheckbox("chkDilate", Potree.settings.dilateEnabled, 
			(checkbox) => {Potree.settings.dilateEnabled = checkbox.checked;}
		);

		hookCheckbox("chkEDL", Potree.settings.edlEnabled, 
			(checkbox) => {Potree.settings.edlEnabled = checkbox.checked;}
		);

		hookCheckbox("chkHQS", Potree.settings.hqsEnabled, 
			(checkbox) => {Potree.settings.hqsEnabled = checkbox.checked;}
		);

		hookCheckbox("chkBoundingBox", Potree.settings.showBoundingBox, 
			(checkbox) => {Potree.settings.showBoundingBox = checkbox.checked;}
		);

		hookCheckbox("chkUpdate", Potree.settings.updateEnabled, 
			(checkbox) => {Potree.settings.updateEnabled = checkbox.checked;}
		);

	}

	


	let panel = new Panel();
	panel.element = elPanel;

	return panel;
}