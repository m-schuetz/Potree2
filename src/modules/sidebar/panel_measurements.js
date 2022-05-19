
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

export function createMeasurementsPanel(){
	let elPanel = document.createElement("div");
	elPanel.id = "measurements_panel";

	let elTitle = document.createElement("div");
	elTitle.classList.add("subsection");
	elTitle.textContent = "Measurements";

	elPanel.append(elTitle);

	{
		let elButton = document.createElement("input");
		elButton.classList.add("potree_sidebar_button");
		elButton.type = "button";
		elButton.title = "Point";
		elButton.style.backgroundImage = `url(${dir}/icons/point.svg)`;

		elButton.addEventListener("click", () => {
			potree.measure.startMeasuring({maxMarkers: 1});
		});

		elPanel.append(elButton);
	}

	{
		let elButton = document.createElement("input");
		elButton.classList.add("potree_sidebar_button");
		elButton.type = "button";
		elButton.title = "Distance";
		elButton.style.backgroundImage = `url(${dir}/icons/distance.svg)`;

		elButton.addEventListener("click", () => {
			potree.measure.startMeasuring({});
		});

		elPanel.append(elButton);
	}

	{
		let elButton = document.createElement("input");
		elButton.classList.add("potree_sidebar_button");
		elButton.type = "button";
		elButton.title = "Height";
		elButton.style.backgroundImage = `url(${dir}/icons/height.svg)`;

		elButton.addEventListener("click", () => {
			potree.measure.startMeasuring({});
		});

		elPanel.append(elButton);
	}

	{
		let elButton = document.createElement("input");
		elButton.classList.add("potree_sidebar_button");
		elButton.type = "button";
		elButton.title = "Circle";
		elButton.style.backgroundImage = `url(${dir}/icons/circle.svg)`;

		elButton.addEventListener("click", () => {
			potree.measure.startMeasuring({});
		});

		elPanel.append(elButton);
	}


	let panel = new Panel();
	panel.element = elPanel;

	return panel;
}