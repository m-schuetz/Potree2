
import {Gradients, Utils} from "potree";
let dir = new URL(import.meta.url + "/../").href;

class Panel{

	constructor(){
		this.element = document.createElement("div");
		this.elTable = document.createElement("table");

		let elTitle = document.createElement("div");
		elTitle.classList.add("subsection");
		elTitle.textContent = "Infos";

		let elButtons = document.createElement("div");
		
		let createButton = (value, onClick) => {
			let elButton = document.createElement("input");

			elButton.type = "button";
			elButton.value = value;
			elButton.onclick = onClick;


			elButtons.append(elButton);
		};

		createButton("📋 camera", () => {
			
			let {camPos, camTarget} = Potree.state;

			let str = `[${camPos}], \n[${camTarget}]`;

			Utils.clipboardCopy(str);
		});

		this.element.append(elTitle);
		this.element.append(this.elTable);
		this.element.append(elButtons);
	}

	set(pointcloud){
		// connect attributes
	}

	update(){

		let valueRows = "";

		valueRows += `
			<tr>
				<td>rendered points</td>
				<td>${Potree.state.numPoints.toLocaleString()}</td>
			</tr><tr>
				<td>rendered voxels</td>
				<td>${Potree.state.numVoxels.toLocaleString()}</td>
			</tr><tr>
			</tr><tr>
				<td>rendered octree nodes</td>
				<td>${Potree.state.numNodes.toLocaleString()}</td>
			</tr><tr>
			</tr><tr>
				<td># 3D Tile Nodes</td>
				<td>${Potree.state.num3DTileNodes.toLocaleString()}</td>
			</tr><tr>
			</tr><tr>
				<td>3D Tile Triangles</td>
				<td>${Potree.state.num3DTileTriangles.toLocaleString()}</td>
			</tr><tr>
				<td>FPS</td>
				<td>${Potree.state.fps}</td>
			</tr><tr>
				<td>cam.pos</td>
				<td>${Potree.state.camPos}</td>
			</tr><tr>
				<td>cam.dir</td>
				<td>${Potree.state.camDir}</td>
			</tr><tr>
				<td>cam.target</td>
				<td>${Potree.state.camTarget}</td>
			</tr>
		`;

		let strTable = `
		<table>
			${valueRows}
		</table>
		`;

		this.elTable.innerHTML = strTable;

		requestAnimationFrame(this.update.bind(this));
	}

}

export function createPanel(){
	let panel = new Panel();
	panel.element.id = "info_panel";
	panel.element.classList.add("subsection_panel");

	panel.update();

	return panel;
}