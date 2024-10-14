
import {Gradients, Utils} from "potree";
import {PointMeasure, DistanceMeasure, HeightMeasure} from "potree";
let dir = new URL(import.meta.url + "/../").href;

function createMarkerTable(measure, prefix = ""){
	let htmlMarkers = "";

	htmlMarkers += `
		<thead>
			<tr>
				<th style="text-align: left;  width: 10%;"></th>
				<th style="text-align: right; width: 30%;">x</th>
				<th style="text-align: right; width: 30%;">y</th>
				<th style="text-align: right; width: 30%;">z</th>
			</tr>
		</thead>
	`;

	for(let i = 0; i < measure.markers.length; i++){

		let marker = measure.markers[i];

		htmlMarkers += `
			<tr id="${prefix}_${i}">
				<td style="text-align: left">${i + 1}</td>
				<td style="text-align: right">${marker.x.toFixed(3)}</td>
				<td style="text-align: right">${marker.y.toFixed(3)}</td>
				<td style="text-align: right">${marker.z.toFixed(3)}</td>
			</tr>
		`;

	}

	let html = `
	<table style="width: 100%">
		${htmlMarkers}
	</table>
	`;
	
	return html;
}

function createDistanceTable(measure){

	let html = `<table>`;

	html += `
		<thead>
			<tr>
				<th style="text-align: left;  width: 10%;"></th>
				<th style="text-align: left;  width: 30%;"></th>
				<th style="text-align: right; width: 30%;">distance</th>
				<th style="text-align: right; width: 30%;">total</th>
			</tr>
		</thead>
	`;

	let total = 0;
	for(let i = 0; i < measure.markers.length; i++){

		let distance = 0;

		if(i > 0){
			let a = measure.markers[i - 1];
			let b = measure.markers[i];

			distance = b.distanceTo(a);
			total = total + distance;
		}

		html += `
			<tr>
				<td style="text-align: left">${i + 1}</td>
				<td style="text-align: left"></td>
				<td style="text-align: right">${distance.toFixed(3)}</td>
				<td style="text-align: right">${total.toFixed(3)}</td>
			</tr>
		`;
	}

	html += "</table>";

	return html;

}


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
						<span name="test" style="justify-self: stretch; font-size: 1.3em;"><b>${measure.label}</b></span>
						<span>(${measure.constructor.name})</span>
					</div>

					${createMarkerTable(measure, `${prefix}`)}
			`;

			if(measure instanceof DistanceMeasure){
				html += `
					${createDistanceTable(measure)}
				`;
			}

			if(measure instanceof HeightMeasure){
				let height = 0;
				if(measure.markers.length === 2){
					height = Math.abs(measure.markers[1].z - measure.markers[0].z);
				}
				html += `
					height: ${height.toFixed(3)}
				`;
			}

			html += `
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