var express = require('express')
var app = express()

const port = 3000

app.use((req, res, next) => {
    console.log(`${new Date()} - ${req.method} request for ${req.url}`)
    next()
})

app.use(express.static('../static'))

app.listen(port, () => {
    console.log(`[server] listening on port ${port}`)
})