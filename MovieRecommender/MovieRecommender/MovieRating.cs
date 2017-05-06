using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MovieRecommender
{
    public class MovieRating
    {
        public int userId { get; set; }
        public int movieId { get; set; }
        public double rating { get; set; }
        public int date { get; set; }

        public MovieRating(int userId, int movieId, double rating, int date)
        {
            this.userId = userId;
            this.movieId = movieId;
            this.rating = rating;
            this.date = date;
        }

        public override string ToString()
        {
            return string.Format("{0}, {1}, {2}, {3}", userId, movieId, rating, date);
        }
    }
}
