
import {Gradients, Utils} from "potree";
let dir = new URL(import.meta.url + "/../").href;

class Panel{

	constructor(){
		this.element = document.createElement("div");
		this.elTable = document.createElement("table");

		let elTitle = document.createElement("div");
		elTitle.classList.add("subsection");
		
		elTitle.textContent = "Hovered";

		this.element.append(elTitle);
		this.element.append(this.elTable);

		this.oldHtmlString = "";
	}

	set(pointcloud){
		// connect attributes
	}

	update(){

		let valueRows = "";

		let hoveredItem = Potree.hoveredItem;

		valueRows += `
		<tr>
			<td>Type</td>
			<td class="table_attributes_td">${hoveredItem?.type ?? "none"}</td>
		</tr>
		`;

		if(hoveredItem?.type === "PointCloudOctreeNode (Point)"){

			let strPos = hoveredItem.position.toString(1);
			let node = hoveredItem.instance;
			let octree = node.octree;

			// let point = octree.getPoint(node, hoveredItem.pointIndex);
			let point = node.getPoint(hoveredItem.pointIndex);

			if(point){

				let strPos = point.position.toArray().map(v => v.toFixed(2)).join(", ");

				valueRows += `
				<tr>
					<td>octree.name</td>
					<td class="table_attributes_td">${octree.name}</td>
				</tr>
				<tr>
					<td>node.name</td>
					<td class="table_attributes_td">${node.name}</td>
				</tr>
				<tr>
					<td>point index</td>
					<td class="table_attributes_td">${hoveredItem.pointIndex}</td>
				</tr>
				<tr>
					<td>position</td>
					<td class="table_attributes_td">${strPos}</td>
				</tr>
				<tr>
					<td colspan="2" style="text-align: center">- Raw Buffer Data -</td>
				</tr>
				<!--<tr>
					<td>point</td>
					<td class="table_attributes_td">${JSON.stringify(point, null, "<br> ")}</td>
				</tr>-->
				<!--<tr>
					<td >position</td>
					<td class="table_attributes_td">${strPos}</td>
				</tr>-->
				`;


				let attributes = octree.loader.attributes;
				if(point.attributes)
				for(let i = 0; i < attributes.attributes.length; i++){
					let attribute = attributes.attributes[i];

					let attributeValues = point.attributes[attribute.name];

					if(!attributeValues) continue;
					
					let str = attributeValues.join(", ");

					valueRows += `
						<tr>
							<td >${attribute.name}</td>
							<td class="table_attributes_td">${str}</td>
						</tr>
					`;

				}
				// let attributes = octree.loader.attributes;
				// for(let i = 0; i < attributes.attributes.length; i++){
				// 	let attribute = attributes.attributes[i];

				// 	let attributeValues = point[attribute.name]

				// 	valueRows += `
				// 		<tr>
				// 			<td >${attribute.name}</td>
				// 			<td class="table_attributes_td">${attributeValues.strValue}</td>
				// 		</tr>
				// 	`;
				// }
			}
		}else if(hoveredItem?.type === "Image360"){
			let {image, images} = hoveredItem;
			let strPos = images.position.clone().add(image.position).toString(1);

			valueRows += `
			<tr>
				<td>name</td>
				<td class="table_attributes_td">${image.name}</td>
			</tr>
			<tr>
				<td>position</td>
				<td class="table_attributes_td">${strPos}</td>
			</tr>
			`;
		}

		let strTable = `
		<style>
			.table_attributes_td{
				width: 100%;
				max-width: 0;
				white-space: nowrap;
				overflow: hidden;
				text-overflow: ellipsis;
			}
		</style>
		
			${valueRows}
		
		`;

		let newHtmlString = strTable;
		if(newHtmlString !== this.oldHtmlString){
			this.elTable.innerHTML = strTable;
			this.elTable.style.width = "100%";
			this.oldHtmlString = newHtmlString;
		}

		requestAnimationFrame(this.update.bind(this));
	}

}

export function createPanel(){
	let panel = new Panel();
	panel.element.id = "hovered_panel";
	panel.element.classList.add("subsection_panel");

	panel.update();

	return panel;
}