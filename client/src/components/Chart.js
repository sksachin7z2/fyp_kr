import React from 'react'
import {Bar,Pie,Line} from 'react-chartjs-2'
import {Chart as ChartJS } from 'chart.js/auto'
function BarChart({chartData}) {
  return (
    <div>
        <Line data={chartData} />
    </div>
  )
}

export default BarChart