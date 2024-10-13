
import {Gradients, Utils} from "potree";
import {PointMeasure, DistanceMeasure, HeightMeasure} from "potree";
let dir = new URL(import.meta.url + "/../").href;

class Panel{

	constructor(){
		this.element = document.createElement("div");
		this.name = "Measurements";
		this.prevHtml = "";

		this.element = document.createElement("div");
		this.element.id = "measurements_panel";
		this.element.classList.add("subsection_panel");

		let elTitle = document.createElement("div");
		elTitle.classList.add("subsection");
		elTitle.textContent = "Measurement Tools";

		this.element.append(elTitle);

		this.createActions();

		let elMeasureTitle = document.createElement("div");
		elMeasureTitle.classList.add("subsection");
		elMeasureTitle.textContent = "Measurements";
		
		this.elMeasures = document.createElement("div");
		this.element.append(elMeasureTitle);
		this.element.append(this.elMeasures);
	}

	set(pointcloud){
		// connect attributes
	}



	createActions(){
		{
			let elButton = document.createElement("input");
			elButton.classList.add("potree_sidebar_button");
			elButton.type = "button";
			elButton.title = "Point";
			elButton.style.backgroundImage = `url(${dir}/icons/point.svg)`;

			elButton.addEventListener("click", () => {
				potree.measure.startMeasuring(new PointMeasure());
			});

			this.element.append(elButton);
		}

		{
			let elButton = document.createElement("input");
			elButton.classList.add("potree_sidebar_button");
			elButton.type = "button";
			elButton.title = "Distance";
			elButton.style.backgroundImage = `url(${dir}/icons/distance.svg)`;

			elButton.addEventListener("click", () => {
				potree.measure.startMeasuring(new DistanceMeasure());
			});

			this.element.append(elButton);
		}

		{
			let elButton = document.createElement("input");
			elButton.classList.add("potree_sidebar_button");
			elButton.type = "button";
			elButton.title = "Height";
			elButton.style.backgroundImage = `url(${dir}/icons/height.svg)`;

			elButton.addEventListener("click", () => {
				potree.measure.startMeasuring(new HeightMeasure());
			});

			this.element.append(elButton);
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

			this.element.append(elButton);
		}
	}

	updateListOfMeasures(){

		let measureTool = potree.measure;

		
		
		let html = "";

		let i = 0;

		for(let measure of measureTool.measures){

			let prefix = `measure_${i}`;

			html += `
				<div>
					<div style="display: grid; grid-template-columns: 1fr 0.1fr" >
						<span name="test" style="justify-self: stretch"><b>${measure.label}</b></span>
						<span>(${measure.constructor.name})</span>
					</div>

					${measure.toHtml(`${prefix}`)}
				</div>
				<br>
			`;

			i++;
		}

		// only update DOM if the html string changed
		if(html != this.prevHtml){
			this.elMeasures.innerHTML = html;
			// this.elMeasures.onmouseover = () => {
			// 	console.log("test");
			// };
			this.prevHtml = html;

			for(let i = 0; i < measureTool.measures.length; i++){
				let measure = measureTool.measures[i];
				let prefix = `measure_${i}`;

				for(let markerIndex = 0; markerIndex < measure.markers.length; markerIndex++){
					let el = this.elMeasures.querySelector(`#${prefix}_${markerIndex}`);

					if(el){
						el.onmouseenter = () => {
							measure.markers_highlighted[markerIndex] = true;
							console.log("enter!");
						};

						el.onmouseleave = () => {
							measure.markers_highlighted[markerIndex] = false;
							console.log("leavbe!");
						};
					}
				}

			}
		}

		requestAnimationFrame(this.updateListOfMeasures.bind(this));

	}

}



export function createMeasurementsPanel(){
	

	// updateListOfMeasures(elMeasures);

	let panel = new Panel();

	panel.updateListOfMeasures();

	return panel;
}